import pandas as pd
import numpy as np
import os
import random

# Конфигурация
CACHE_DIR = "./cache_cg_flg_sale_flg_2024-05-31"
OUTPUT_FILE = f"{CACHE_DIR}/graph_ready_dataset_pd_full_parallel.parquet"
N_SAMPLES = 620
SEED = 42
NUM_FEATURES = 43


np.random.seed(SEED)
random.seed(SEED)

# --- КОНСТРУКТОР ИМЕН ПРИЗНАКОВ ---
# Словарь частей, для генерации похожих, но фейковых имен
PREFIXES = ['dep', 'srv', 'crd', 'prd', 'lne', 'inv', 'pl', 'seg', 'cla', 'pnl']
MIDDLES = ['acct', 'trx', 'otf', 'pos', 'inf', 'sbol', 'payroll', 'income', 'total', 'active', 'risk', 'transfer']
SUFFIXES = ['amt', 'qty', 'cd', 'diff', 'val', 'rate', 'bal', 'flag']


def generate_fake_feature_name(existing_names):
    """Генерирует уникальное имя признака, похожее на реальное."""
    while True:
        p = random.choice(PREFIXES)
        m = random.choice(MIDDLES)
        s = random.choice(SUFFIXES)
        # Иногда добавляем еще одну часть для сложности
        if random.random() > 0.5:
            m2 = random.choice(['mnth', '3m', '12m', 'avg', 'curr', 'hist'])
            name = f"{p}_{m}_{m2}_{s}"
        else:
            name = f"{p}_{m}_{s}"

        if name not in existing_names:
            return name


def generate_synthetic_data():
    print(f"Генерация {N_SAMPLES} строк синтетических данных...")

    df = pd.DataFrame()

    # 1. Обязательные поля (как в ТЗ)
    df['epk_id'] = np.arange(N_SAMPLES).astype(str)
    df['report_dt'] = np.random.choice(['2025-02-28', '2025-03-31', '2025-04-30'], N_SAMPLES)

    # Технические поля (для совместимости)
    df['actual_epk_id'] = df['epk_id']
    df['row_upd_dttm'] = '2025-05-01'
    df['ctl_loading'] = 1
    df['score_uplift'] = np.random.uniform(0, 1, N_SAMPLES)
    # df['sdo_pass_0m'] = np.random.uniform(0, 10000, N_SAMPLES)

    # 2. Генерация Causal Structure (Treatment -> Mediator -> Target)
    # Мы генерируем скрытые непрерывные величины, чтобы создать корреляцию,
    # а затем бинаризуем их, чтобы получить int (0/1).

    # Скрытый конфаундер (например, "Активность клиента")
    hidden_activity = np.random.normal(0, 1, N_SAMPLES)

    # Treatment: comm_fl (INT)
    # Вероятность коммуникации зависит от активности
    # Используем порог для бинаризации
    lat_comm = 0.6 * hidden_activity + np.random.normal(0, 1, N_SAMPLES)
    df['comm_fl'] = (lat_comm > 0.2).astype(np.int32)  # 0 или 1

    # Mediator: sale_fl (INT)
    # Зависит от comm_fl (сильная связь) и активности
    # Если была коммуникация (comm_fl=1), вероятность продажи выше
    lat_sale = 1.5 * df['comm_fl'] + 0.4 * hidden_activity + np.random.normal(0, 1, N_SAMPLES)
    df['sale_fl'] = (lat_sale > 1.0).astype(np.int32)  # 0 или 1

    # Target: sdo_pass_5m (FLOAT)
    # Зависит от sale_fl (сильно), comm_fl (слабо) и активности
    # Это деньги, поэтому делаем значения большими и положительными (через exp или abs)
    base_sdo = 5000 * df['sale_fl'] + 1000 * df['comm_fl'] + 2000 * hidden_activity + np.random.normal(0, 2000,
                                                                                                       N_SAMPLES)
    df['sdo_pass_5m'] = np.abs(base_sdo + 10000).astype(np.float32)


    # 3. Генерация остальных num_features признаков (Фейковые имена)
    # Генерируем num_features случайных имени
    fake_features = []
    reserved_names = list(df.columns)
    for _ in range(NUM_FEATURES):
        name = generate_fake_feature_name(reserved_names + fake_features)
        fake_features.append(name)

    print(f"Сгенерировано {len(fake_features)} фейковых признаков.")

    # Заполняем их данными
    for col in fake_features:
        # Случайный тип данных: либо float (деньги/скоры), либо int (количества/флаги)
        if 'amt' in col or 'val' in col or 'rate' in col:
            # Float с корреляцией к таргету (чтобы граф не был пустым)
            df[col] = (0.1 * df['sdo_pass_5m'] + np.random.normal(0, 1000, N_SAMPLES)).astype(np.float32)
        else:
            # Int (например, количество транзакций)
            df[col] = np.random.poisson(5, N_SAMPLES).astype(np.int32)

    # 4. Финальная сборка и сохранение
    # Порядок колонок не важен для паркета, но важен состав
    final_cols = ['epk_id', 'report_dt', 'sdo_pass_5m', 'comm_fl', 'sale_fl'] + fake_features


    df_final = df[final_cols].copy()

    os.makedirs(CACHE_DIR, exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE)

    print(f"Файл сохранен: {OUTPUT_FILE}")
    print(f"Размер: {df_final.shape}")
    print("\nТипы данных (проверка):")
    print(df_final[['comm_fl', 'sale_fl', 'sdo_pass_5m']].dtypes)
    print("\nПример данных:")
    print(df_final[['comm_fl', 'sale_fl', 'sdo_pass_5m']].head())


if __name__ == "__main__":
    generate_synthetic_data()