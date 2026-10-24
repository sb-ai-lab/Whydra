"""
Генератор синтетических банковских данных v3 (Sber Hardcore).

Цель: Создать датасет, на котором алгоритм PC (с тестом Fisher-Z) будет
работать ЧАСАМИ, а не секундами, имитируя реальные данные Сбера.

Как достигается вычислительная сложность:
1. Плотный истинный DAG: Каждый узел имеет от 5 до 15 родителей.
2. Ненормальные распределения: Экспоненциальные (деньги), Пуассоновские (штуки).
3. Zero-inflation: 40% значений принудительно зануляются (клиент не пользуется).
4. Артефакт fillna(-1): Блоки признаков одновременно принимают значение -1
   (имитация отсутствия продукта).
5. Ordinal Encoding: Категориальные фичи симулируются как случайные int.

Из-за этих искажений линейная частичная корреляция (Fisher-Z) никогда не
падает до идеального нуля. При N=600,000 даже остаточный шум корреляции 0.02
дает p-value < 0.01. PC не может удалить ребра, граф остается плотным,
возникает комбинаторный взрыв (миллионы тестов на глубинах 4, 5, 6).

Использование:
    python data_generator_v3_hardcore.py --n_rows 600000 --n_features 47
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.special import expit

# Реальные названия из вашего ноутбука
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


def generate_hardcore_data(n_rows: int = 600_000, n_features: int = 47, seed: int = 42,
                           output_dir: str = "./synthetic_data"):
    rng = np.random.RandomState(seed)

    # 1. Формируем список колонок
    key_cols = ['comm_fl', 'sale_fl', 'sdo_pass_5m']
    bg_cols = REAL_FEATURE_NAMES.copy()

    # Добиваем или обрезаем до нужного количества
    while len(bg_cols) + len(key_cols) < n_features:
        bg_cols.append(f"extra_feature_{len(bg_cols)}")
    bg_cols = bg_cols[:n_features - len(key_cols)]

    all_cols = key_cols + bg_cols

    # 2. Строим плотный DAG (топологическая сортировка)
    # Каждый узел зависит от 5 до 15 предыдущих узлов
    dag = {col: [] for col in all_cols}
    data_raw = np.zeros((n_rows, n_features))

    print(f"Generating base dense DAG for {n_features} features...")
    for i in range(n_features):
        col_name = all_cols[i]

        # Генерируем базовый шум
        node_data = rng.normal(0, 1, n_rows)

        if i > 0:
            # Выбираем от 5 до 15 родителей из предыдущих узлов
            n_parents = min(i, rng.randint(5, 16))
            parents_idx = rng.choice(range(i), size=n_parents, replace=False)

            for p_idx in parents_idx:
                p_name = all_cols[p_idx]
                dag[col_name].append(p_name)

                # Вес связи (сильная корреляция)
                weight = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
                node_data += weight * data_raw[:, p_idx]

        # Нормализуем, чтобы дисперсия не улетала в космос
        data_raw[:, i] = node_data / (np.std(node_data) + 1e-5)

    # 3. Применяем жесткие нелинейности (имитация реальных данных)
    print("Applying non-linearities, heavy tails, and zero-inflation...")
    df = pd.DataFrame(data_raw, columns=all_cols)

    for col in all_cols:
        vals = df[col].values

        if col in ['comm_fl', 'sale_fl', 'prl_tax_payer_inn']:
            # Бинарные флаги
            df[col] = (expit(vals) > 0.5).astype(np.int32)

        elif 'amt' in col or 'bal' in col or 'val' in col or 'sdo' in col:
            # Деньги: Экспоненциальное распределение (тяжелый правый хвост)
            # Сдвигаем и берем экспоненту
            money = np.exp(vals * 1.5) * 1000
            # Zero-inflation: 40% клиентов имеют 0 на счету
            zero_mask = rng.rand(n_rows) < 0.4
            money[zero_mask] = 0
            df[col] = money.astype(np.float32)

        elif 'qty' in col or 'diff' in col or 'mnth' in col:
            # Штуки/Дни: Пуассоновское распределение
            counts = np.clip(vals * 5 + 5, 0, 100)
            # Zero-inflation: 30% нулей
            zero_mask = rng.rand(n_rows) < 0.3
            counts[zero_mask] = 0
            df[col] = np.round(counts).astype(np.int32)

        elif 'cd' in col:
            # Категориальные (Ordinal Encoded) - от 0 до 15
            cats = np.clip(np.round(vals * 3 + 7), 0, 15)
            df[col] = cats.astype(np.int32)

        else:
            # Остальные (скоры, рейты)
            df[col] = (expit(vals) * 100).astype(np.float32)

    # 4. Имитация артефакта fillna(-1)
    # В реальных данных, если у клиента нет кредитки, все поля по кредитке равны -1
    print("Injecting fillna(-1) artifacts (correlated missingness)...")

    # Создаем 4 "продукта" (группы колонок)
    product_blocks = [[c for c in all_cols if 'crd' in c],  # Кредитки
                      [c for c in all_cols if 'dep' in c],
                      # Депозиты[c for c in all_cols if 'lne' in c], # Кредиты наличными[c for c in all_cols if 'pnl' in c], # PnL
                      ]

    for block in product_blocks:
        if len(block) < 2: continue
        # 30% клиентов не имеют этого продукта
        missing_mask = rng.rand(n_rows) < 0.3
        for col in block:
            # Заменяем значения на -1
            df.loc[missing_mask, col] = -1.0

    # Добавляем технические колонки
    df.insert(0, "epk_id", np.arange(n_rows).astype(str))
    df.insert(1, "report_dt", rng.choice(["2025-02-28", "2025-03-31", "2025-04-30"], n_rows))

    # 5. Сохранение
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, f"synth_banking_v3_{n_features}f.parquet")
    dag_path = os.path.join(output_dir, f"synth_banking_v3_{n_features}f_dag.txt")

    df.to_parquet(parquet_path)

    n_edges = sum(len(v) for v in dag.values())
    with open(dag_path, "w") as f:
        for child, parents in dag.items():
            for p in parents:
                f.write(f"{p} -> {child}\n")

    print(f"\n=== SUCCESS ===")
    print(f"Dataset saved to: {parquet_path}")
    print(f"DAG saved to: {dag_path} ({n_edges} edges)")

    # Вывод статистики для проверки
    print("\nData sample (showing non-linearities and -1s):")
    print(
        df[['comm_fl', 'sale_fl', 'sdo_pass_5m', 'crd_dc_mnth_snc_open_qty', 'dep_acct_dep_tot_bal_rub_amt']].head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=600_000)
    parser.add_argument("--n_features", type=int, default=47)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    args = parser.parse_args()

    generate_hardcore_data(args.n_rows, args.n_features, args.seed, args.output_dir)