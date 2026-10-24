import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import argparse
import os


def fix_data_for_fisherz(input_path: str, output_path: str):
    print(f"Загрузка данных из: {input_path}")
    df = pd.read_parquet(input_path)

    # Исключаем технические колонки из трансформации
    tech_cols = ['epk_id', 'report_dt', 'actual_epk_id', 'row_upd_dttm', 'ctl_loading', 'score_uplift']
    tech_df = df[[c for c in tech_cols if c in df.columns]].copy()

    # Колонки для трансформации
    causal_cols = [c for c in df.columns if c not in tech_cols]
    causal_df = df[causal_cols].copy()

    print(f"Обработка {len(causal_cols)} признаков...")

    # 1. Лечим артефакт fillna(-1)
    # Заменяем -1 на случайные значения из этой же колонки (среди тех, что не -1)
    # Это разрушит искусственную корреляцию между пропусками
    for col in causal_cols:
        # Предполагаем, что -1.0 или -1 это пропуск
        mask_missing = (causal_df[col] == -1) | (causal_df[col] == -1.0)
        n_missing = mask_missing.sum()

        if n_missing > 0:
            # Берем значения, которые НЕ равны -1
            valid_values = causal_df.loc[~mask_missing, col].values
            if len(valid_values) > 0:
                # Сэмплируем случайные значения для заполнения
                random_fills = np.random.choice(valid_values, size=n_missing, replace=True)
                causal_df.loc[mask_missing, col] = random_fills

    # 2. Гауссианизация (Quantile Transformation)
    # Превращаем любые распределения в нормальное (Gaussian)
    print("Применение QuantileTransformer (Гауссианизация)...")
    qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=10000)

    # Трансформируем
    transformed_vals = qt.fit_transform(causal_df)
    causal_df = pd.DataFrame(transformed_vals, columns=causal_cols, index=causal_df.index)

    # 3. Собираем обратно
    final_df = pd.concat([tech_df, causal_df], axis=1)

    # Сохраняем
    final_df.to_parquet(output_path)
    print(f"Готово! Исправленные данные сохранены в: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Путь к исходному паркету")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения исправленного паркета")
    args = parser.parse_args()

    fix_data_for_fisherz(args.input, args.output)