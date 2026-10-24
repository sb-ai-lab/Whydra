"""
Продвинутый генератор синтетических банковских данных для Causal Discovery.

Генерирует данные с:
- Реалистичными именами признаков из банковского домена
- Контролируемой каузальной структурой (DAG)
- Настраиваемым характером функциональных зависимостей (линейные / квази-линейные / нелинейные)
- Параметризуемым количеством признаков и строк
- Сохранением в Parquet-формат

Использование:
    python data_generator_advanced.py --n_rows 500000 --n_features 30 --nonlinearity linear --seed 42
    python data_generator_advanced.py --n_rows 500000 --n_features 30 --nonlinearity quasi --seed 42
    python data_generator_advanced.py --n_rows 500000 --n_features 30 --nonlinearity nonlinear --seed 42

Параметр nonlinearity:
    - "linear"    : 100% линейные зависимости
    - "quasi"     : 80-90% линейные, остальное — слабые нелинейные слагаемые
    - "nonlinear" : < 50% линейные, существенные нелинейные компоненты (log, quadratic, interaction)
"""

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit  # сигмоида


# ============================================================
#  РЕАЛИСТИЧНЫЕ БАНКОВСКИЕ ИМЕНА ПРИЗНАКОВ
# ============================================================

# Взяты с фотографий реальных данных + расширены
REAL_FEATURE_NAMES: List[str] = [
    # --- Депозиты ---
    "dep_acct_dep_mnth_lst_clsr_qty",
    "dep_acct_dep_max_close_dt_diff",
    "dep_acct_dep_td_qty",
    "dep_acct_dep_d_bal_rub_amt",
    "dep_acct_dep_nd_bal_rub_amt",
    "dep_acct_tot_davg_qrt_rub_amt",
    "dep_acct_dep_tot_1000_qty",
    # --- Кредиты ---
    "crd_dc_tot_dl_rub_amt",
    "crd_trx_dc_cash_3m_qty",
    "crd_trx_dc_dsh_3m_amt",
    "crd_trx_spnd_cash_avg_3m_amt",
    "crd_pos_home_repair_rub_3m_amt",
    "crd_dc_pos_leisure_rub_amt",
    "crd_dc_pos_clothes_rub_amt",
    "crd_otf_trnf_org_rub_amt",
    "crd_inf_transfer_rub_amt",
    "crd_txn_spnd_3m_amt",
    # --- Продукты ---
    "prd_lst_prod_cd",
    "prd_dep_active_qty",
    "prd_ma_active_nflag",
    # --- Сервисы ---
    "srv_sbol_txn_other_3m_amt",
    "srv_sbol_mob_1st_login_ever_dt_diff",
    "srv_sbol_txn_mob_3m_amt",
    "srv_sbol_mob_1m_amt",
    "srv_pmt_dc_amt",
    # --- Инвестиции ---
    "inv_tot_br_rub_amt",
    # --- Страхование ---
    "insur_total_bal",
    # --- Сегменты и классы ---
    "seg_client_criteria",
    "cla_dep_active_last_dt_diff",
    "cla_mg_active_last_dt_diff",
    # --- Доходы / PnL ---
    "dep_inf_payroll_rub_amt",
    "dep_inf_income_qty",
    "pnl_oi_ca_amt",
    "dep_pens_pln_bal_rub_amt",
    # --- Возраст / профиль ---
    "sd_age_frac_qty",
    "pl_12m_cltv",
    "prl_tax_payer_inn",
    "dep_payroll_lgnt_tot_rub_amt",
    # --- Дополнительные ---
    "dep_off_total_rub_amt",
    "crd_off_offr_rub_amt",
]


def _generate_extra_names(n: int, existing: set) -> List[str]:
    """Генерирует дополнительные уникальные банковские имена признаков."""
    prefixes = ["dep", "srv", "crd", "prd", "lne", "inv", "pl", "seg", "cla", "pnl"]
    middles = [
        "acct", "trx", "otf", "pos", "inf", "sbol", "payroll", "income",
        "total", "active", "risk", "transfer", "txn", "spnd", "pmt",
    ]
    suffixes = ["amt", "qty", "cd", "diff", "val", "rate", "bal", "flag"]
    modifiers = ["mnth", "3m", "12m", "avg", "curr", "hist", "1m", "6m", "tot"]

    result = []
    attempts = 0
    while len(result) < n and attempts < n * 50:
        attempts += 1
        p = random.choice(prefixes)
        m = random.choice(middles)
        s = random.choice(suffixes)
        if random.random() > 0.4:
            mod = random.choice(modifiers)
            name = f"{p}_{m}_{mod}_{s}"
        else:
            name = f"{p}_{m}_{s}"
        if name not in existing and name not in result:
            result.append(name)
    return result


# ============================================================
#  КАУЗАЛЬНАЯ СТРУКТУРА (DAG)
# ============================================================

def build_causal_dag(
    feature_names: List[str],
    target_col: str = "sdo_pass_5m",
    treatment_col: str = "comm_fl",
    mediator_col: str = "sale_fl",
    edge_density: float = 0.08,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Строит DAG (словарь: узел -> список родителей).

    Гарантирует каузальную цепочку: comm_fl -> sale_fl -> sdo_pass_5m
    и дополнительные связи между фоновыми фичами и ключевыми узлами.
    """
    rng = np.random.RandomState(seed)

    # Все признаки кроме ключевых
    background = [f for f in feature_names if f not in [target_col, treatment_col, mediator_col]]
    rng.shuffle(background)

    dag: Dict[str, List[str]] = {name: [] for name in feature_names}

    # Ключевая цепочка
    # comm_fl зависит от части background (конфаундеры)
    n_confounders = max(3, int(len(background) * 0.1))
    confounders = background[:n_confounders]
    dag[treatment_col] = confounders[:3]

    # sale_fl зависит от comm_fl и части конфаундеров
    dag[mediator_col] = [treatment_col] + confounders[:2]

    # sdo_pass_5m зависит от sale_fl, comm_fl (слабо), и части фичей
    dag[target_col] = [mediator_col, treatment_col] + confounders[:2] + background[n_confounders:n_confounders + 3]

    # Связи между background фичами (имитация реальной корреляционной структуры)
    # Порядок = топологический порядок
    for i, feat in enumerate(background):
        if i == 0:
            continue
        # Каждая фича зависит от 0-3 предыдущих фич
        n_parents = rng.choice([0, 1, 2, 3], p=[0.3, 0.35, 0.25, 0.1])
        if n_parents > 0 and i > 0:
            possible_parents = background[:i]
            n_parents = min(n_parents, len(possible_parents))
            parents = list(rng.choice(possible_parents, size=n_parents, replace=False))
            dag[feat] = parents

    # Дополнительные ребра от background к target/mediator/treatment для реализма
    extra_features_to_target = background[n_confounders + 3:n_confounders + 8]
    for f in extra_features_to_target:
        if f not in dag[target_col]:
            dag[target_col].append(f)

    return dag


# ============================================================
#  ГЕНЕРАЦИЯ ДАННЫХ ПО DAG
# ============================================================

def _apply_function(
    parent_values: np.ndarray,
    coeff: float,
    nonlinearity: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Применяет функциональную зависимость от родителя.
    
    nonlinearity:
        "linear"    - y = coeff * x
        "quasi"     - y = coeff * x + small_nonlinear_term
        "nonlinear" - y = coeff * f(x), где f — нелинейная функция
    """
    x = parent_values
    
    if nonlinearity == "linear":
        return coeff * x

    elif nonlinearity == "quasi":
        # 80-90% линейный сигнал + слабый нелинейный
        nl_type = rng.choice(["sq", "log", "interaction"])
        nl_coeff = coeff * rng.uniform(0.05, 0.15)  # слабый нелинейный коэффициент
        linear_part = coeff * x
        if nl_type == "sq":
            return linear_part + nl_coeff * (x ** 2) / (np.std(x) + 1e-6)
        elif nl_type == "log":
            return linear_part + nl_coeff * np.log1p(np.abs(x))
        else:
            return linear_part + nl_coeff * np.sin(x / (np.std(x) + 1e-6))

    elif nonlinearity == "nonlinear":
        # Существенные нелинейные зависимости
        nl_type = rng.choice(["quadratic", "log_transform", "sigmoid", "interaction", "exp_decay"])
        
        if nl_type == "quadratic":
            return coeff * x + coeff * 0.5 * (x ** 2) / (np.std(x) + 1e-6)
        elif nl_type == "log_transform":
            return coeff * np.sign(x) * np.log1p(np.abs(x))
        elif nl_type == "sigmoid":
            return coeff * 10 * (expit(x / (np.std(x) + 1e-6)) - 0.5)
        elif nl_type == "interaction":
            return coeff * np.sin(2 * x / (np.std(x) + 1e-6)) * np.abs(x) ** 0.5
        else:  # exp_decay
            return coeff * x * np.exp(-np.abs(x) / (3 * np.std(x) + 1e-6))
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")


def generate_data_from_dag(
    dag: Dict[str, List[str]],
    feature_names: List[str],
    n_rows: int,
    nonlinearity: str = "linear",
    target_col: str = "sdo_pass_5m",
    treatment_col: str = "comm_fl",
    mediator_col: str = "sale_fl",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует данные по DAG с заданным типом нелинейности.
    """
    rng = np.random.RandomState(seed)
    data = {}

    # Определяем типы фичей (по суффиксам)
    def infer_type(name: str) -> str:
        if name in [treatment_col, mediator_col]:
            return "binary"
        if name == target_col:
            return "positive_float"
        if any(name.endswith(s) for s in ["_flag", "_nflag", "_fl", "_cd"]):
            return "discrete_small"
        if any(name.endswith(s) for s in ["_amt", "_val", "_bal", "_rate"]):
            return "float"
        if any(name.endswith(s) for s in ["_qty", "_diff"]):
            return "integer"
        return "float"

    # Топологическая сортировка DAG
    in_degree = {n: 0 for n in feature_names}
    for node, parents in dag.items():
        if node in in_degree:
            in_degree[node] = len(parents)

    topo_order = []
    queue = [n for n in feature_names if in_degree[n] == 0]
    visited = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        topo_order.append(node)
        for child, parents in dag.items():
            if node in parents and child not in visited:
                in_degree[child] -= 1
                if in_degree[child] <= 0:
                    queue.append(child)

    # Дополняем оставшиеся (циклические или забытые)
    for n in feature_names:
        if n not in topo_order:
            topo_order.append(n)

    # Скрытый конфаундер — «активность клиента»
    hidden_activity = rng.normal(0, 1, n_rows)

    # Генерируем данные в топологическом порядке
    for name in topo_order:
        parents = dag.get(name, [])
        ftype = infer_type(name)
        noise_scale = rng.uniform(0.5, 2.0)

        # Базовый шум
        if ftype in ("float", "positive_float"):
            noise = rng.normal(0, noise_scale * 1000, n_rows)
        elif ftype == "integer":
            noise = rng.normal(0, noise_scale * 5, n_rows)
        elif ftype in ("binary", "discrete_small"):
            noise = rng.normal(0, noise_scale, n_rows)
        else:
            noise = rng.normal(0, noise_scale * 100, n_rows)

        # Сумма вкладов родителей
        signal = np.zeros(n_rows)
        for p in parents:
            if p not in data:
                continue
            coeff = rng.uniform(0.2, 1.5) * rng.choice([-1, 1])

            # Для ключевых связей — усиленные коэффициенты
            if name == mediator_col and p == treatment_col:
                coeff = rng.uniform(1.5, 2.5)
            elif name == target_col and p == mediator_col:
                coeff = rng.uniform(2.0, 4.0)
            elif name == target_col and p == treatment_col:
                coeff = rng.uniform(0.5, 1.0)

            parent_data = data[p].astype(np.float64)
            contribution = _apply_function(parent_data, coeff, nonlinearity, rng)
            signal += contribution

        # Добавляем влияние скрытого конфаундера для некоторых фич
        if rng.random() < 0.3:
            signal += rng.uniform(0.1, 0.5) * hidden_activity * (1000 if ftype == "float" else 1)

        raw = signal + noise

        # Приведение к нужному типу
        if name == treatment_col:
            # Бинарный: comm_fl
            prob = expit(0.5 * hidden_activity + rng.normal(0, 0.5, n_rows))
            for p in parents:
                if p in data:
                    prob += 0.1 * expit(data[p].astype(np.float64) / (np.std(data[p]) + 1e-6))
            prob = np.clip(prob / (1 + 0.1 * len(parents)), 0.05, 0.95)
            data[name] = rng.binomial(1, prob).astype(np.int32)

        elif name == mediator_col:
            # Бинарный: sale_fl  (зависит от comm_fl через signal)
            latent = signal + 0.3 * hidden_activity + rng.normal(0, 1, n_rows)
            data[name] = (latent > np.median(latent)).astype(np.int32)

        elif name == target_col:
            # Положительный float: sdo_pass_5m (деньги)
            base = 5000 * data.get(mediator_col, np.zeros(n_rows)).astype(np.float64)
            base += 1000 * data.get(treatment_col, np.zeros(n_rows)).astype(np.float64)
            base += 2000 * hidden_activity
            base += signal * 0.1
            base += rng.normal(0, 2000, n_rows)
            data[name] = np.abs(base + 10000).astype(np.float32)

        elif ftype == "binary" or ftype == "discrete_small":
            if ftype == "binary":
                threshold = np.percentile(raw, rng.uniform(40, 70))
                data[name] = (raw > threshold).astype(np.int32)
            else:
                # Дискретное значение 0..N
                n_vals = rng.choice([2, 3, 5, 10])
                data[name] = np.clip(
                    np.digitize(raw, np.linspace(raw.min(), raw.max(), n_vals + 1)[1:-1]),
                    0, n_vals - 1,
                ).astype(np.int32)

        elif ftype == "integer":
            # Количества (положительные целые)
            data[name] = np.clip(np.round(np.abs(raw)).astype(int), 0, 100000).astype(np.int32)

        elif ftype == "positive_float":
            data[name] = np.abs(raw).astype(np.float32)

        else:
            data[name] = raw.astype(np.float32)

    df = pd.DataFrame(data)
    return df


# ============================================================
#  ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def generate_synthetic_banking_data(
    n_rows: int = 500_000,
    n_features: int = 30,
    nonlinearity: str = "linear",
    seed: int = 42,
    output_dir: str = "./synthetic_data",
    edge_density: float = 0.08,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Основная функция генерации.
    
    Args:
        n_rows: количество строк (400_000 — 600_000)
        n_features: количество признаков (20 — 60)
        nonlinearity: "linear" | "quasi" | "nonlinear"
        seed: random seed
        output_dir: директория для сохранения
        edge_density: плотность ребер в DAG
    
    Returns:
        (DataFrame, DAG-словарь)
    """
    np.random.seed(seed)
    random.seed(seed)

    # 1. Собираем имена признаков
    key_features = ["sdo_pass_5m", "comm_fl", "sale_fl"]
    n_background = n_features - len(key_features)
    assert n_background > 0, f"n_features ({n_features}) must be > {len(key_features)}"

    # Берем реальные имена, если хватает, иначе догенерим
    available_real = [n for n in REAL_FEATURE_NAMES if n not in key_features]
    if len(available_real) >= n_background:
        background_names = available_real[:n_background]
    else:
        background_names = list(available_real)
        extra = _generate_extra_names(
            n_background - len(background_names),
            set(background_names + key_features),
        )
        background_names += extra

    all_features = key_features + background_names
    print(f"[Generator] Features: {len(all_features)} (key=3, background={len(background_names)})")
    print(f"[Generator] Rows: {n_rows}")
    print(f"[Generator] Nonlinearity: {nonlinearity}")

    # 2. Строим DAG
    dag = build_causal_dag(
        feature_names=all_features,
        edge_density=edge_density,
        seed=seed,
    )
    n_edges = sum(len(v) for v in dag.values())
    print(f"[Generator] DAG: {len(all_features)} nodes, {n_edges} edges")

    # 3. Генерируем данные
    df = generate_data_from_dag(
        dag=dag,
        feature_names=all_features,
        n_rows=n_rows,
        nonlinearity=nonlinearity,
        seed=seed,
    )

    # 4. Добавляем технические поля
    df.insert(0, "epk_id", np.arange(n_rows).astype(str))
    df.insert(1, "report_dt", np.random.choice(
        ["2025-02-28", "2025-03-31", "2025-04-30"], n_rows
    ))

    # 5. Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{n_features}f_{nonlinearity}"
    parquet_path = os.path.join(output_dir, f"synth_banking_{tag}.parquet")
    dag_path = os.path.join(output_dir, f"synth_banking_{tag}_dag.txt")

    # Убираем технические поля из парквета для графа
    cols_for_graph = [c for c in df.columns if c not in ["epk_id", "report_dt"]]
    df[cols_for_graph].to_parquet(parquet_path)
    print(f"[Generator] Saved data: {parquet_path} ({df.shape})")

    # Сохраняем DAG для валидации метрик
    with open(dag_path, "w") as f:
        for node, parents in dag.items():
            if parents:
                for p in parents:
                    f.write(f"{p} -> {node}\n")
    print(f"[Generator] Saved DAG: {dag_path}")

    return df, dag


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced synthetic banking data generator")
    parser.add_argument("--n_rows", type=int, default=500_000, help="Number of rows (default: 500000)")
    parser.add_argument("--n_features", type=int, default=30, help="Number of features (default: 30)")
    parser.add_argument("--nonlinearity", type=str, default="linear",
                        choices=["linear", "quasi", "nonlinear"],
                        help="Type of functional dependencies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    args = parser.parse_args()

    df, dag = generate_synthetic_banking_data(
        n_rows=args.n_rows,
        n_features=args.n_features,
        nonlinearity=args.nonlinearity,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print(f"\n=== Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nKey columns dtypes:")
    for c in ["comm_fl", "sale_fl", "sdo_pass_5m"]:
        print(f"  {c}: {df[c].dtype}, sample = {df[c].head(3).tolist()}")
    print(f"\nFirst 5 rows (key columns):")
    print(df[["comm_fl", "sale_fl", "sdo_pass_5m"]].head())


if __name__ == "__main__":
    main()
