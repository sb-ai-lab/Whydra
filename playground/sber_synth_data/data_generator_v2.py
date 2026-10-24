"""
Генератор синтетических банковских данных v2 — приближенный к реальным данным Сбера.

Ключевые отличия от v1:
═══════════════════════════════════════════════════════════════════════════
Проблема v1: PC на синтетике работал за 14 сек, на реальных данных — 2.8 часа.
Причина: v1 генерировал РАЗРЕЖЕННУЮ матрицу корреляций (каждая фича зависит 
от 0-3 родителей), а реальные банковские данные имеют ПЛОТНУЮ корреляцию.

Что делает v2:
1. ЛАТЕНТНЫЕ ФАКТОРЫ — все фичи генерируются из 5-8 скрытых факторов (активность
   клиента, доход, кредитная нагрузка, возраст, цифровая активность и т.д.),
   что создаёт групповые корреляции между фичами одного домена.
2. МУЛЬТИКОЛЛИНЕАРНОСТЬ — внутри каждой тематической группы (депозиты, кредиты, 
   сервисы) фичи сильно коррелированы между собой.
3. ПРОПУСКИ → -1 — как в реальных данных Сбера (dataset.fillna(-1)), что создаёт
   дополнительные артефактные корреляции через сентинельное значение.
4. alpha=0.01 — строгий порог (как на реальных данных), из-за чего рёбра 
   удаляются медленнее → max_degree остаётся высоким → экспоненциально больше 
   CI-тестов на каждой глубине.
5. ПЛОТНЫЙ ГРАФ — correlation matrix близка к полносвязной на начальных глубинах,
   что резко увеличивает C(max_neighbors, depth) и время работы PC.
═══════════════════════════════════════════════════════════════════════════

Использование:
    python data_generator_v2.py --n_rows 600000 --n_features 47 --seed 42
    python data_generator_v2.py --n_rows 600000 --n_features 47 --nonlinearity quasi --seed 42
"""

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit


# ============================================================
#  РЕАЛИСТИЧНЫЕ БАНКОВСКИЕ ИМЕНА (с фото реальных данных)
# ============================================================

FEATURE_POOL: List[dict] = [
    # --- Депозиты (группа: deposit) ---
    {"name": "dep_acct_dep_mnth_lst_clsr_qty", "group": "deposit", "type": "int", "miss": 0.15},
    {"name": "dep_acct_dep_max_close_dt_diff", "group": "deposit", "type": "int", "miss": 0.20},
    {"name": "dep_acct_dep_td_qty",            "group": "deposit", "type": "int", "miss": 0.05},
    {"name": "dep_acct_dep_d_bal_rub_amt",     "group": "deposit", "type": "float", "miss": 0.10},
    {"name": "dep_acct_dep_nd_bal_rub_amt",    "group": "deposit", "type": "float", "miss": 0.10},
    {"name": "dep_acct_tot_davg_qrt_rub_amt",  "group": "deposit", "type": "float", "miss": 0.12},
    {"name": "dep_acct_dep_tot_1000_qty",      "group": "deposit", "type": "int", "miss": 0.08},
    {"name": "dep_acct_dc_bal_rub_amt",        "group": "deposit", "type": "float", "miss": 0.10},
    {"name": "dep_off_total_rub_amt",          "group": "deposit", "type": "float", "miss": 0.25},
    {"name": "dep_inf_payroll_rub_amt",        "group": "income", "type": "float", "miss": 0.30},
    {"name": "dep_inf_income_qty",             "group": "income", "type": "int", "miss": 0.20},
    {"name": "dep_pens_pln_bal_rub_amt",       "group": "deposit", "type": "float", "miss": 0.40},
    {"name": "dep_payroll_lgnt_tot_rub_amt",   "group": "income", "type": "float", "miss": 0.35},
    # --- Кредиты (группа: credit) ---
    {"name": "crd_dc_tot_dl_rub_amt",          "group": "credit", "type": "float", "miss": 0.15},
    {"name": "crd_trx_dc_cash_3m_qty",         "group": "credit_trx", "type": "int", "miss": 0.10},
    {"name": "crd_trx_dc_dsh_3m_amt",          "group": "credit_trx", "type": "float", "miss": 0.10},
    {"name": "crd_trx_spnd_cash_avg_3m_amt",   "group": "credit_trx", "type": "float", "miss": 0.12},
    {"name": "crd_pos_home_repair_rub_3m_amt",  "group": "credit", "type": "float", "miss": 0.60},
    {"name": "crd_dc_pos_leisure_rub_amt",     "group": "credit", "type": "float", "miss": 0.45},
    {"name": "crd_dc_pos_clothes_rub_amt",     "group": "credit", "type": "float", "miss": 0.50},
    {"name": "crd_otf_trnf_org_rub_amt",       "group": "credit", "type": "float", "miss": 0.35},
    {"name": "crd_inf_transfer_rub_amt",       "group": "credit", "type": "float", "miss": 0.20},
    {"name": "crd_txn_spnd_3m_amt",            "group": "credit_trx", "type": "float", "miss": 0.08},
    {"name": "crd_trx_dc_cash_3m_amt",         "group": "credit_trx", "type": "float", "miss": 0.10},
    # --- Продукты (группа: product) ---
    {"name": "prd_lst_prod_cd",                "group": "product", "type": "discrete", "miss": 0.05},
    {"name": "prd_dep_active_qty",             "group": "product", "type": "int", "miss": 0.03},
    {"name": "prd_ma_active_nflag",            "group": "product", "type": "binary", "miss": 0.02},
    # --- Сервисы (группа: service) ---
    {"name": "srv_sbol_txn_other_3m_amt",      "group": "service", "type": "float", "miss": 0.15},
    {"name": "srv_sbol_mob_1st_login_ever_dt_diff", "group": "service", "type": "int", "miss": 0.25},
    {"name": "srv_sbol_txn_mob_3m_amt",        "group": "service", "type": "float", "miss": 0.15},
    {"name": "srv_sbol_mob_1m_amt",            "group": "service", "type": "float", "miss": 0.20},
    {"name": "srv_pmt_dc_amt",                 "group": "service", "type": "float", "miss": 0.10},
    # --- Инвестиции / Страхование ---
    {"name": "inv_tot_br_rub_amt",             "group": "invest", "type": "float", "miss": 0.50},
    {"name": "insur_total_bal",                "group": "invest", "type": "float", "miss": 0.55},
    # --- Сегменты / профиль ---
    {"name": "seg_client_criteria",            "group": "profile", "type": "discrete", "miss": 0.03},
    {"name": "cla_dep_active_last_dt_diff",    "group": "profile", "type": "int", "miss": 0.15},
    {"name": "cla_mg_active_last_dt_diff",     "group": "profile", "type": "int", "miss": 0.30},
    {"name": "sd_age_frac_qty",                "group": "profile", "type": "float", "miss": 0.02},
    {"name": "prl_tax_payer_inn",              "group": "profile", "type": "binary", "miss": 0.10},
    # --- PnL ---
    {"name": "pnl_oi_ca_amt",                  "group": "pnl", "type": "float", "miss": 0.20},
    {"name": "crd_off_offr_rub_amt",           "group": "credit", "type": "float", "miss": 0.40},
    # --- Кредитные линии ---
    {"name": "lne_curr_pmt_ann_mg_rub_amt",    "group": "credit", "type": "float", "miss": 0.30},
    {"name": "lne_pl_open_wavg_intr_rate",     "group": "credit", "type": "float", "miss": 0.35},
    {"name": "lne_min_exp_pl_ann_dt_diff",     "group": "credit", "type": "int", "miss": 0.35},
    {"name": "lne_lst_open_pl_rate",           "group": "credit", "type": "float", "miss": 0.35},
]


def _extra_name(existing: set, rng) -> str:
    prefixes = ["dep", "srv", "crd", "prd", "lne", "inv", "pl", "seg", "cla", "pnl"]
    middles = ["acct", "trx", "otf", "pos", "inf", "sbol", "payroll", "income", "total", "active", "risk", "transfer"]
    suffixes = ["amt", "qty", "cd", "diff", "val", "rate", "bal", "flag"]
    mods = ["mnth", "3m", "12m", "avg", "curr", "hist", "1m", "6m", "tot"]
    for _ in range(500):
        p = rng.choice(prefixes)
        m = rng.choice(middles)
        s = rng.choice(suffixes)
        mod = rng.choice(mods) if rng.random() > 0.4 else ""
        name = f"{p}_{m}_{mod}_{s}" if mod else f"{p}_{m}_{s}"
        if name not in existing:
            return name
    return f"feat_{rng.randint(0, 99999)}"


# ============================================================
#  ЛАТЕНТНЫЕ ФАКТОРЫ (скрытые переменные)
# ============================================================

LATENT_FACTOR_NAMES = [
    "client_activity",      # общая активность клиента
    "income_level",         # уровень дохода
    "credit_load",          # кредитная нагрузка
    "digital_engagement",   # цифровая вовлечённость (сбол, моб)
    "age_maturity",         # возраст / зрелость отношений с банком
    "investment_propensity", # склонность к инвестициям
    "spending_pattern",     # паттерн расходов
    "risk_appetite",        # аппетит к риску
]

# Какие группы фичей зависят от каких латентных факторов
GROUP_TO_FACTORS = {
    "deposit":    ["income_level", "client_activity", "age_maturity"],
    "income":     ["income_level", "client_activity"],
    "credit":     ["credit_load", "income_level", "risk_appetite"],
    "credit_trx": ["credit_load", "spending_pattern", "digital_engagement"],
    "product":    ["client_activity", "age_maturity"],
    "service":    ["digital_engagement", "client_activity"],
    "invest":     ["investment_propensity", "income_level", "risk_appetite"],
    "profile":    ["age_maturity", "income_level"],
    "pnl":        ["income_level", "credit_load", "client_activity"],
}


# ============================================================
#  ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================

def generate_banking_data_v2(
    n_rows: int = 600_000,
    n_features: int = 47,
    nonlinearity: str = "linear",
    seed: int = 42,
    output_dir: str = "./synthetic_data",
    missing_rate_multiplier: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Генерирует синтетические банковские данные с плотной корреляционной структурой.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)
    np.random.seed(seed)

    # --- 1. Подбор фичей ---
    key_features = ["sdo_pass_5m", "comm_fl", "sale_fl"]
    n_background = n_features - len(key_features)

    pool = FEATURE_POOL.copy()
    rng.shuffle(pool)
    
    if len(pool) >= n_background:
        selected_pool = pool[:n_background]
    else:
        selected_pool = pool.copy()
        existing = set(f["name"] for f in selected_pool) | set(key_features)
        while len(selected_pool) < n_background:
            name = _extra_name(existing, rng)
            existing.add(name)
            group = rng.choice(list(GROUP_TO_FACTORS.keys()))
            selected_pool.append({
                "name": name, "group": group,
                "type": rng.choice(["float", "int", "float", "float"]),
                "miss": rng.uniform(0.05, 0.40),
            })

    feature_names = key_features + [f["name"] for f in selected_pool]
    feature_meta = {f["name"]: f for f in selected_pool}
    
    print(f"[GenV2] Features: {len(feature_names)} (key=3, background={n_background})")
    print(f"[GenV2] Rows: {n_rows}, Nonlinearity: {nonlinearity}")

    # --- 2. Генерация латентных факторов ---
    n_latent = len(LATENT_FACTOR_NAMES)
    # Коррелированные латентные факторы (через Cholesky)
    latent_corr = np.eye(n_latent)
    # Добавляем корреляции между факторами
    cross_corrs = [
        (0, 1, 0.4),   # activity ~ income
        (1, 2, -0.3),  # income ~ credit_load (отрицательно)
        (0, 3, 0.5),   # activity ~ digital
        (4, 1, 0.3),   # age ~ income
        (5, 1, 0.4),   # invest ~ income
        (6, 2, 0.3),   # spending ~ credit
        (7, 2, 0.5),   # risk ~ credit
    ]
    for i, j, corr in cross_corrs:
        if i < n_latent and j < n_latent:
            latent_corr[i, j] = latent_corr[j, i] = corr

    # Обеспечиваем положительную определённость
    eigvals = np.linalg.eigvalsh(latent_corr)
    if eigvals.min() < 0.01:
        latent_corr += np.eye(n_latent) * (0.02 - eigvals.min())

    L = np.linalg.cholesky(latent_corr)
    Z = rng.normal(0, 1, (n_rows, n_latent))
    latent = Z @ L.T  # (n_rows, n_latent)
    latent_dict = {name: latent[:, i] for i, name in enumerate(LATENT_FACTOR_NAMES)}

    print(f"[GenV2] Latent factors: {n_latent}, cross-correlations: {len(cross_corrs)}")

    # --- 3. Генерация фоновых фичей из латентных факторов ---
    data = {}
    dag: Dict[str, List[str]] = {name: [] for name in feature_names}

    for feat_info in selected_pool:
        fname = feat_info["name"]
        fgroup = feat_info["group"]
        ftype = feat_info["type"]
        miss_rate = feat_info["miss"] * missing_rate_multiplier

        # Базовый сигнал из латентных факторов
        factors = GROUP_TO_FACTORS.get(fgroup, ["client_activity"])
        signal = np.zeros(n_rows)
        for factor_name in factors:
            if factor_name in latent_dict:
                coeff = rng.uniform(0.3, 1.5) * rng.choice([-1, 1])
                factor_vals = latent_dict[factor_name]
                signal += _apply_nl(factor_vals, coeff, nonlinearity, rng)

        # Зависимость от 1-3 ранее сгенерированных фичей (внутригрупповые корреляции)
        same_group = [n for n in data.keys() if n in feature_meta
                      and feature_meta[n]["group"] == fgroup]
        if same_group:
            n_deps = min(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]), len(same_group))
            parents = list(rng.choice(same_group, size=n_deps, replace=False))
            dag[fname] = parents
            for p in parents:
                coeff = rng.uniform(0.1, 0.6)
                signal += _apply_nl(data[p].astype(np.float64), coeff, nonlinearity, rng)

        # Зависимость от фичей другой группы (межгрупповые корреляции, слабые)
        other_group = [n for n in data.keys() if n in feature_meta
                       and feature_meta[n]["group"] != fgroup]
        if other_group and rng.random() < 0.3:
            p = rng.choice(other_group)
            dag[fname].append(p)
            coeff = rng.uniform(0.05, 0.2)
            signal += _apply_nl(data[p].astype(np.float64), coeff, nonlinearity, rng)

        # Шум
        noise_scale = rng.uniform(0.3, 1.0)  # Меньше шума → более плотные корреляции
        if ftype == "float":
            raw = signal * 5000 + rng.normal(0, noise_scale * 3000, n_rows)
            vals = raw.astype(np.float32)
        elif ftype == "int":
            raw = signal * 10 + rng.normal(0, noise_scale * 5, n_rows)
            vals = np.clip(np.round(np.abs(raw)).astype(int), 0, 50000).astype(np.int32)
        elif ftype == "binary":
            prob = expit(signal)
            vals = rng.binomial(1, prob).astype(np.int32)
        elif ftype == "discrete":
            n_cats = rng.choice([3, 5, 8, 12])
            vals = np.clip(np.digitize(signal, np.linspace(signal.min(), signal.max(), n_cats + 1)[1:-1]),
                           0, n_cats - 1).astype(np.int32)
        else:
            raw = signal * 1000 + rng.normal(0, noise_scale * 500, n_rows)
            vals = raw.astype(np.float32)

        # Пропуски → заменяем на -1 (как в реальных данных Сбера: dataset.fillna(-1))
        if miss_rate > 0:
            # Пропуски коррелированы с латентными факторами (не random!)
            miss_factor = latent_dict.get(factors[0], rng.normal(0, 1, n_rows))
            miss_prob = expit(-2 + miss_rate * 6 + 0.5 * miss_factor)
            miss_mask = rng.binomial(1, miss_prob).astype(bool)
            vals = vals.astype(np.float64)
            vals[miss_mask] = -1.0
            vals = vals.astype(np.float32) if ftype == "float" else vals.astype(np.int32)

        data[fname] = vals

    # --- 4. Ключевые переменные ---
    activity = latent_dict["client_activity"]
    income = latent_dict["income_level"]

    # Treatment: comm_fl (отправка SMS)
    comm_signal = 0.5 * activity + 0.3 * income + rng.normal(0, 0.8, n_rows)
    # Зависит от нескольких фоновых фичей
    bg_names = list(data.keys())[:5]
    for p in bg_names[:3]:
        dag["comm_fl"].append(p)
        comm_signal += 0.1 * data[p].astype(np.float64) / (np.std(data[p]) + 1e-6)
    data["comm_fl"] = (comm_signal > np.percentile(comm_signal, 50)).astype(np.int32)

    # Mediator: sale_fl (открытие кредита)
    sale_signal = 1.5 * data["comm_fl"] + 0.4 * activity + 0.3 * income
    dag["sale_fl"] = ["comm_fl"] + bg_names[:2]
    for p in bg_names[:2]:
        sale_signal += 0.15 * data[p].astype(np.float64) / (np.std(data[p]) + 1e-6)
    sale_signal += rng.normal(0, 1, n_rows)
    data["sale_fl"] = (sale_signal > np.median(sale_signal)).astype(np.int32)

    # Target: sdo_pass_5m
    target_signal = (5000 * data["sale_fl"]
                     + 1000 * data["comm_fl"]
                     + 2000 * income
                     + 1500 * activity)
    dag["sdo_pass_5m"] = ["sale_fl", "comm_fl"] + bg_names[:5]
    for p in bg_names[:5]:
        target_signal += 0.05 * data[p].astype(np.float64)
    target_signal += rng.normal(0, 2000, n_rows)
    data["sdo_pass_5m"] = np.abs(target_signal + 10000).astype(np.float32)

    # --- 5. Сборка DataFrame ---
    df = pd.DataFrame(data, columns=feature_names)

    # Добавляем технические поля
    df.insert(0, "epk_id", np.arange(n_rows).astype(str))
    df.insert(1, "report_dt", rng.choice(["2025-02-28", "2025-03-31", "2025-04-30"], n_rows))

    # --- 6. Проверка плотности корреляций ---
    numeric_cols = [c for c in feature_names if c not in ["epk_id", "report_dt"]]
    corr_matrix = df[numeric_cols].astype(float).corr().values
    np.fill_diagonal(corr_matrix, 0)
    n_significant_corrs = np.sum(np.abs(corr_matrix) > 0.05) // 2
    n_total_pairs = len(numeric_cols) * (len(numeric_cols) - 1) // 2
    density = n_significant_corrs / n_total_pairs

    print(f"[GenV2] Correlation density (|r|>0.05): {density:.1%} ({n_significant_corrs}/{n_total_pairs} pairs)")
    print(f"[GenV2] Mean |correlation|: {np.mean(np.abs(corr_matrix)):.4f}")
    print(f"[GenV2] Max |correlation|: {np.max(np.abs(corr_matrix)):.4f}")

    # --- 7. Сохранение ---
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{n_features}f_{nonlinearity}"
    parquet_path = os.path.join(output_dir, f"synth_banking_v2_{tag}.parquet")
    dag_path = os.path.join(output_dir, f"synth_banking_v2_{tag}_dag.txt")

    cols_for_graph = [c for c in df.columns if c not in ["epk_id", "report_dt"]]
    df[cols_for_graph].to_parquet(parquet_path)
    print(f"[GenV2] Saved: {parquet_path} ({df.shape})")

    n_dag_edges = sum(len(v) for v in dag.values())
    with open(dag_path, "w") as f:
        for node, parents in dag.items():
            for p in parents:
                f.write(f"{p} -> {node}\n")
    print(f"[GenV2] DAG: {n_dag_edges} edges, saved: {dag_path}")

    return df, dag


def _apply_nl(x: np.ndarray, coeff: float, mode: str, rng: np.random.RandomState) -> np.ndarray:
    """Применяет зависимость с контролируемой нелинейностью."""
    if mode == "linear":
        return coeff * x
    elif mode == "quasi":
        nl = coeff * rng.uniform(0.05, 0.15)
        return coeff * x + nl * np.sign(x) * np.log1p(np.abs(x))
    elif mode == "nonlinear":
        t = rng.choice(["sq", "sig", "log"])
        if t == "sq":
            return coeff * x + coeff * 0.3 * x**2 / (np.std(x) + 1e-6)
        elif t == "sig":
            return coeff * 5 * (expit(x / (np.std(x) + 1e-6)) - 0.5)
        else:
            return coeff * np.sign(x) * np.log1p(np.abs(x))
    return coeff * x


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Banking synthetic data generator v2 (dense correlations)")
    parser.add_argument("--n_rows", type=int, default=600_000)
    parser.add_argument("--n_features", type=int, default=47)
    parser.add_argument("--nonlinearity", type=str, default="linear",
                        choices=["linear", "quasi", "nonlinear"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    parser.add_argument("--missing_rate_multiplier", type=float, default=1.0,
                        help="Multiplier for missing rate (>1 = more missings = denser correlations)")
    args = parser.parse_args()

    df, dag = generate_banking_data_v2(
        n_rows=args.n_rows,
        n_features=args.n_features,
        nonlinearity=args.nonlinearity,
        seed=args.seed,
        output_dir=args.output_dir,
        missing_rate_multiplier=args.missing_rate_multiplier,
    )

    print(f"\n=== Summary ===")
    print(f"Shape: {df.shape}")
    for c in ["comm_fl", "sale_fl", "sdo_pass_5m"]:
        col = df[c]
        print(f"  {c}: dtype={col.dtype}, mean={col.mean():.2f}, std={col.std():.2f}")
    
    # Показываем количество -1 (пропусков)
    numeric = df.select_dtypes(include=[np.number])
    n_minus1 = (numeric == -1).sum().sum()
    total_cells = numeric.shape[0] * numeric.shape[1]
    print(f"\n  Sentinel values (-1): {n_minus1:,} / {total_cells:,} ({n_minus1/total_cells:.1%})")


if __name__ == "__main__":
    main()
