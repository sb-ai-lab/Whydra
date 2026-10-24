"""
Генератор синтетических банковских данных v4 (Sber Hardcore + Complexity Matrix).

Цель: Создать масштабируемый стенд, имитирующий реальные банковские данные,
на котором можно гибко управлять вычислительной сложностью алгоритмов каузального вывода
(PC, FCI, RAI с тестом Fisher-Z) и тестировать их пределы.

Как в v4 достигается "поломка" тестов и комбинаторный взрыв:

1. Зоопарк нелинейностей (--nl_types): Тест Fisher-Z ищет ТОЛЬКО линейные связи (Пирсон).
   - Монотонные искажения (log, exp, cube, sig) оставляют "шумовые" корреляции. Тест видит
     корреляцию 0.05, и при N=600,000 строк p-value падает < 0.01. Алгоритм "боится"
     удалять ложные ребра.
   - Симметричные/периодические (abs, sin, cos) дают корреляцию Пирсона ~0. Истинные связи
     полностью скрываются от алгоритма, заставляя его строить ложные обходные пути.

2. Управляемая плотность (--density): В режиме `dense` узел имеет 5-15 родителей. Из-за
   того, что нелинейности (п.1) не дают удалить ребра на малых глубинах (Depth=0, 1),
   алгоритм доходит до Depth=4, 5, 6 с огромным числом соседей. Начинается комбинаторный
   взрыв (например, С(15, 5) = 3003 проверки независимости только для одной пары узлов).

3. Тяжелые хвосты: Банковские суммы (деньги) генерируются с экстремальными хвостами,
   а штуки (транзакции) — с дискретными пуассоновскими скачками. Это грубо нарушает
   базовое требование Fisher-Z о многомерном нормальном распределении.

4. Zero-inflation (Зануления): До 40% значений принудительно обнуляются (имитация
   "клиент не пользуется"). Это создает огромные пики в нуле, ломающие любую
   линейную аппроксимацию связи между признаками.

5. Артефакт fillna(-1): Целые блоки признаков (например, все переменные по кредитке)
   одновременно принимают значение -1, если у клиента нет продукта. Это создает мощнейшие
   искусственные линейные корреляции (клики). Fisher-Z видит 100% синхронное изменение
   и принципиально не может разорвать эти связи.

Поддерживает матрицу приоритетов:
--nonlinearity:
    linear: 100% связей линейны (Y = w*X)
    quasi: ~15% нелинейных связей
    nonlinear: ~70% нелинейных связей

--nl_types (Типы используемых нелинейностей):
    Можно передать список: --nl_types sq log sin
    Или использовать все сразу: --nl_types mixed
    Доступны: sq, cube, log, exp, sig, sin, cos, tanh, abs, step

--density:
    sparse: 1-4 родителя у узла (реалистичный граф)
    dense: 5-15 родителей (экстремально плотный граф)

"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.special import expit

# ============================================================
# РЕЕСТР НЕЛИНЕЙНЫХ ФУНКЦИЙ
# (Нормализованы, чтобы избежать вычислительных взрывов)
# ============================================================
NONLINEAR_FUNCTIONS = {
    # Монотонные
    "sq": lambda x, w: w * np.sign(x) * (np.abs(x) ** 2) / (np.std(x) + 1e-5),
    "cube": lambda x, w: w * (x ** 3) / (np.std(x ** 3) + 1e-5),
    "log": lambda x, w: w * np.sign(x) * np.log1p(np.abs(x)) * 2.0,
    "exp": lambda x, w: w * np.sign(x) * (np.exp(np.clip(np.abs(x), 0, 3)) - 1) / 3.0,
    "sig": lambda x, w: w * 5 * (expit(x) - 0.5),
    "tanh": lambda x, w: w * 3 * np.tanh(x),

    # Немонотонные (самые сложные для каузального вывода)
    "sin": lambda x, w: w * 2 * np.sin(x * 2.0),
    "cos": lambda x, w: w * 2 * np.cos(x * 2.0),
    "abs": lambda x, w: w * (np.abs(x) - np.mean(np.abs(x))),  # центрированный модуль
    "step": lambda x, w: w * 2 * np.where(x > 0, 1.0, -1.0)
}

REAL_FEATURE_NAMES = [
    'lne_curr_pmt_ann_mg_rub_amt', 'dep_acct_dep_tot_bal_rub_amt', 'dep_topup_12m_avg_rub_amt',
    'residual_npv_val_lag_6', 'dep_otf_wthdr_rub_amt', 'lne_pl_debt_os_rub_amt', 'pnl_oi_pl_amt',
    'crd_dc_mnth_snc_open_qty', 'pnl_oi_total_amt', 'lne_lst_open_pl_rate', 'lne_tot_debt_due_rub_amt',
    'crd_pos_auto_rub_3m_amt', 'lne_curr_pmt_ann_pl_rub_amt', 'pl_12m_cltv', 'cltv', 'srv_pmt_utl_amt',
    'crd_dc_pos_auto_rub_amt', 'lne_min_exp_pl_ann_dt_diff', 'crd_otf_fee_rub_amt', 'crd_otf_loan_qty',
    'lne_pl_open_wavg_intr_rate', 'lne_tot_issued_open_rub_amt', 'dep_otf_loan_qty', 'cltv_lag_5',
    'crd_trx_dc_cash_6m_spnd_qty', 'dep_acct_dc_bal_rub_amt', 'prd_yrs_lst_open_qty', 'dep_inc_avg_risk_rub_amt',
    'pnl_oi_cc_amt', 'pnl_oi_ml_amt', 'dep_acct_tot_davg_mnth_rub_amt', 'dep_acct_tot_davg_yr_rub_amt',
    'pl_val_lag_4', 'pl_1m_cltv', 'dep_upr_tot_bal_rub_amt', 'pl_val_lag_5', 'dep_inf_income_rub_amt',
    'lne_act_mg_os_rub_amt', 'pnl_oi_td_amt', 'crd_pos_special_rub_3m_amt', 'lne_mg_max_exp_dt_diff',
    'pnl_oi_ca_amt', 'prd_lst_prod_cd', 'prl_tax_payer_inn'
]


def apply_structural_equation(x: np.ndarray, weight: float, nonlinearity: str,
                              nl_types: list, rng: np.random.RandomState) -> np.ndarray:
    """Применяет математическую функцию связи X -> Y согласно выбранной сложности."""

    if nonlinearity == "linear":
        return weight * x

    # Выбираем случайную функцию из разрешенных
    chosen_nl = rng.choice(nl_types)
    nl_func = NONLINEAR_FUNCTIONS[chosen_nl]

    if nonlinearity == "quasi":
        # 15% шанс на нелинейность, 85% линейно
        if rng.rand() < 0.15:
            return nl_func(x, weight)
        return weight * x

    elif nonlinearity == "nonlinear":
        # 70% шанс на нелинейность, 30% линейно
        if rng.rand() < 0.70:
            return nl_func(x, weight)
        return weight * x


def generate_v4_data(n_rows: int, n_features: int, nonlinearity: str, nl_types: list,
                     density: str, seed: int, output_dir: str):
    rng = np.random.RandomState(seed)

    # Разбор типов нелинейности
    if "mixed" in nl_types or "all" in nl_types:
        active_nl_types = list(NONLINEAR_FUNCTIONS.keys())
    else:
        active_nl_types = [t for t in nl_types if t in NONLINEAR_FUNCTIONS]
        if not active_nl_types:
            print("[WARN] Не найдено валидных типов нелинейностей, используется 'mixed'")
            active_nl_types = list(NONLINEAR_FUNCTIONS.keys())

    print(f"Active non-linear functions: {active_nl_types}")

    key_cols = ['comm_fl', 'sale_fl', 'sdo_pass_5m']
    bg_cols = REAL_FEATURE_NAMES.copy()

    while len(bg_cols) + len(key_cols) < n_features:
        bg_cols.append(f"extra_feature_{len(bg_cols)}")
    bg_cols = bg_cols[:n_features - len(key_cols)]
    all_cols = key_cols + bg_cols

    dag = {col: [] for col in all_cols}
    data_raw = np.zeros((n_rows, n_features))

    print(f"Generating {density} DAG ({nonlinearity} dependencies) for {n_features} features...")

    # 1. Структурные уравнения (Каузальный Скелет)
    for i in range(n_features):
        col_name = all_cols[i]
        node_data = rng.normal(0, 1, n_rows)  # Базовый шум узла

        if i > 0:
            if density == "sparse":
                n_parents = min(i, rng.randint(1, 4))  # 1-3 родителя (реалистично)
            else:
                n_parents = min(i, rng.randint(5, 16))  # 5-15 родителей (Hardcore)

            parents_idx = rng.choice(range(i), size=n_parents, replace=False)

            for p_idx in parents_idx:
                dag[col_name].append(all_cols[p_idx])
                weight = rng.uniform(0.3, 0.9) * rng.choice([-1, 1])

                # Добавляем влияние родителя
                node_data += apply_structural_equation(data_raw[:, p_idx], weight, nonlinearity, active_nl_types, rng)

        data_raw[:, i] = node_data / (np.std(node_data) + 1e-5)

    # 2. Банковские артефакты
    print("Applying banking artifacts (Zero-inflation, heavy tails)...")
    df = pd.DataFrame(data_raw, columns=all_cols)

    for col in all_cols:
        vals = df[col].values

        if col in ['comm_fl', 'sale_fl', 'prl_tax_payer_inn']:
            df[col] = (expit(vals) > 0.5).astype(np.int32)

        elif 'amt' in col or 'bal' in col or 'val' in col or 'sdo' in col:
            money = np.abs(vals * 50000 + rng.normal(0, 10000, n_rows))
            zero_mask = rng.rand(n_rows) < 0.4  # 40% нулей
            money[zero_mask] = 0
            df[col] = money.astype(np.float32)

        elif 'qty' in col or 'diff' in col or 'mnth' in col:
            counts = np.clip(vals * 5 + 5, 0, 100)
            zero_mask = rng.rand(n_rows) < 0.3
            counts[zero_mask] = 0
            df[col] = np.round(counts).astype(np.int32)

        elif 'cd' in col:
            cats = np.clip(np.round(vals * 3 + 7), 0, 15)
            df[col] = cats.astype(np.int32)

        else:
            df[col] = (vals * 100).astype(np.float32)

    # 3. Имитация пропусков -1.0
    print("Injecting fillna(-1) missingness...")
    product_blocks = [[c for c in all_cols if 'crd' in c], [c for c in all_cols if 'dep' in c]
                      ]
    for block in product_blocks:
        if len(block) < 2: continue
        missing_mask = rng.rand(n_rows) < 0.25
        for col in block:
            df.loc[missing_mask, col] = -1.0

    df.insert(0, "epk_id", np.arange(n_rows).astype(str))
    df.insert(1, "report_dt", rng.choice(["2025-02-28", "2025-03-31", "2025-04-30"], n_rows))

    # 4. Сохранение
    os.makedirs(output_dir, exist_ok=True)
    nl_tag = "mixed" if "mixed" in nl_types else "-".join(active_nl_types[:3])
    tag = f"{n_features}f_{nonlinearity}_{nl_tag}_{density}"
    parquet_path = os.path.join(output_dir, f"synth_banking_v4_{tag}.parquet")
    dag_path = os.path.join(output_dir, f"synth_banking_v4_{tag}_dag.txt")

    cols_for_graph = [c for c in df.columns if c not in ["epk_id", "report_dt"]]
    df[cols_for_graph].to_parquet(parquet_path)

    n_edges = sum(len(v) for v in dag.values())
    with open(dag_path, "w") as f:
        for child, parents in dag.items():
            for p in parents:
                f.write(f"{p} -> {child}\n")

    print(f"\n=== SUCCESS ===")
    print(f"Dataset saved to: {parquet_path}")
    print(f"DAG saved to: {dag_path} ({n_edges} edges)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=600_000)
    parser.add_argument("--n_features", type=int, default=47)
    parser.add_argument("--nonlinearity", type=str, default="linear", choices=["linear", "quasi", "nonlinear"])

    # Новый параметр для выбора типов нелинейности
    parser.add_argument("--nl_types", nargs="+", default=["mixed"],
                        help="Типы нелинейностей. Доступны: sq, cube, log, exp, sig, tanh, sin, cos, abs, step, mixed")

    parser.add_argument("--density", type=str, default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    args = parser.parse_args()

    generate_v4_data(args.n_rows, args.n_features, args.nonlinearity, args.nl_types, args.density, args.seed,
                     args.output_dir)