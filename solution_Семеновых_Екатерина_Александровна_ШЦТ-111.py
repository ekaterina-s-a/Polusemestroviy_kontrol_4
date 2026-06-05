import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Конфигурация
CONFIG = {
    'data_path': Path("./data_train"),
    'window_days': 30,            # окно для истории
    'mad_multiplier': 4.5,        # множитель MAD (~3 сигма)
    'min_samples': 10,            # минимальное количество исторических точек
    'min_daily_ots': 1.0,         # минимальный OTS для рассмотрения
    'output_dir': Path('./output'),
    'plots_dir': Path('./output/plots')
}

CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)
CONFIG['plots_dir'].mkdir(exist_ok=True, parents=True)

# Загрузка всех parquet-файлов
def load_all_data(data_path):
    if not data_path.exists():
        raise FileNotFoundError(f"Папка с данными не найдена: {data_path}")
    all_files = list(data_path.glob("month=*/**/*.parquet"))
    print(f"Найдено файлов: {len(all_files)}")
    dfs = []
    for f in all_files:
        df = pd.read_parquet(f)
        # Приведение весов к float
        for col in ['Weight', 'week_weight', 'month_weight', 'BrandinDelivery']:
            if col in df.columns and df[col].dtype.name in ('object', 'decimal'):
                df[col] = df[col].astype(float)
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    if 'researchdate' in df_all.columns:
        df_all['researchdate'] = pd.to_datetime(df_all['researchdate'])
    print(f"Всего строк: {len(df_all)}")
    return df_all

# Агрегация OTS на уровне (SubjectID, дата, CategoryDelivery, Brand)
def prepare_ots_data(df):
    cat_col = 'CategoryNameDelivery' if 'CategoryNameDelivery' in df.columns else 'CategoryDelivery'
    mask = (df['BrandinDelivery'] == 1.0) & (df[cat_col].notna())
    df_filtered = df[mask].copy()
    grouped = df_filtered.groupby(
        ['SubjectID', 'researchdate', cat_col, 'BrandID', 'Brand']
    ).agg(
        count_rows=('QueryText', 'count'),
        weight=('Weight', 'first')
    ).reset_index()
    grouped['daily_ots'] = grouped['weight'] * grouped['count_rows']
    grouped.rename(columns={cat_col: 'CategoryDelivery'}, inplace=True)
    return grouped

# Расчёт доли респондента в дневном OTS бренда
def compute_brand_shares(ots_data):
    brand_daily = ots_data.groupby(
        ['CategoryDelivery', 'BrandID', 'Brand', 'researchdate']
    )['daily_ots'].sum().reset_index()
    brand_daily.rename(columns={'daily_ots': 'total_brand_ots'}, inplace=True)
    merged = ots_data.merge(brand_daily, on=['CategoryDelivery', 'BrandID', 'Brand', 'researchdate'])
    merged['share'] = merged['daily_ots'] / merged['total_brand_ots']
    merged = merged[merged['daily_ots'] >= CONFIG['min_daily_ots']]
    return merged

# Обнаружение аномалий для одного бренда
def detect_anomalies_by_brand(brand_data):
    brand_data = brand_data.sort_values('researchdate').copy()
    anomalies = []
    anomaly_reasons = []

    for _, row in brand_data.iterrows():
        current_date = row['researchdate']
        current_share = row['share']
        subject_id = row['SubjectID']
        brand_id = row['BrandID']
        cat_delivery = row['CategoryDelivery']

        # Исторические данные за window_days дней до текущей даты
        historical = brand_data[
            (brand_data['researchdate'] < current_date) &
            (brand_data['researchdate'] >= current_date - timedelta(days=CONFIG['window_days']))
        ]

        if len(historical) < CONFIG['min_samples']:
            continue

        median_share = historical['share'].median()
        mad = np.median(np.abs(historical['share'] - median_share))

        if mad == 0:
            threshold = median_share + 0.01
        else:
            threshold = median_share + CONFIG['mad_multiplier'] * mad

        # Аномалия: доля выше адаптивного порога (малые OTS уже отсечены min_daily_ots)
        if current_share > threshold:
            anomalies.append((subject_id, current_date))
            anomaly_reasons.append({
                'SubjectID': subject_id,
                'researchdate': current_date,
                'BrandID': brand_id,
                'Brand': row['Brand'],
                'CategoryDelivery': cat_delivery,
                'daily_ots': row['daily_ots'],
                'score': current_share,
                'share': current_share,
                'median_share': median_share,
                'mad': mad,
                'threshold': threshold,
                'reason': f"share={current_share:.4f} > threshold={threshold:.4f}"
            })

    return anomalies, anomaly_reasons

# Запуск детекции по всем брендам
def run_anomaly_detection(shares_data):
    unique_brands = shares_data[['CategoryDelivery', 'BrandID']].drop_duplicates()
    all_anomalies = []
    all_reasons = []

    for _, (cat, brand) in unique_brands.iterrows():
        brand_subset = shares_data[
            (shares_data['CategoryDelivery'] == cat) & (shares_data['BrandID'] == brand)
        ]
        if len(brand_subset) < CONFIG['min_samples']:
            continue
        anoms, reasons = detect_anomalies_by_brand(brand_subset)
        all_anomalies.extend(anoms)
        all_reasons.extend(reasons)

    anomalies_df = pd.DataFrame(all_anomalies, columns=['SubjectID', 'researchdate']).drop_duplicates()
    reasons_df = pd.DataFrame(all_reasons)
    return anomalies_df, reasons_df

# Расчёт общего OTS по дням до и после удаления
def calculate_total_ots(ots_data, anomalies_df):
    df_copy = ots_data.copy()
    anom_copy = anomalies_df.copy()
    anom_copy['key'] = anom_copy['SubjectID'].astype(str) + '_' + anom_copy['researchdate'].astype(str)
    df_copy['key'] = df_copy['SubjectID'].astype(str) + '_' + df_copy['researchdate'].astype(str)
    df_clean = df_copy[~df_copy['key'].isin(anom_copy['key'])].copy()
    ots_before = df_copy.groupby('researchdate')['daily_ots'].sum()
    ots_after = df_clean.groupby('researchdate')['daily_ots'].sum()
    return ots_before, ots_after

# Расчёт изменения OTS по категориям CategoryDelivery
def calculate_category_ots_change(ots_data, anomalies_df):
    df_copy = ots_data.copy()
    anom_copy = anomalies_df.copy()
    anom_copy['key'] = anom_copy['SubjectID'].astype(str) + '_' + anom_copy['researchdate'].astype(str)
    df_copy['key'] = df_copy['SubjectID'].astype(str) + '_' + df_copy['researchdate'].astype(str)
    df_clean = df_copy[~df_copy['key'].isin(anom_copy['key'])].copy()
    cat_before = df_copy.groupby('CategoryDelivery')['daily_ots'].sum()
    cat_after = df_clean.groupby('CategoryDelivery')['daily_ots'].sum()
    cat_pct = (cat_after / cat_before * 100).sort_values()
    return cat_pct

# График изменения OTS по категориальному признаку (пол, возраст, регион, ...)
def plot_profile_ots_change(df_raw, ots_data, anomalies_df, profile_col, title, filename):
    if profile_col not in df_raw.columns:
        print(f"Колонка '{profile_col}' отсутствует, пропуск {filename}")
        return
    anom_copy = anomalies_df.copy()
    anom_copy['key'] = anom_copy['SubjectID'].astype(str) + '_' + anom_copy['researchdate'].astype(str)
    profile_before = df_raw[['SubjectID', 'researchdate', profile_col]].drop_duplicates()
    profile_before['key'] = profile_before['SubjectID'].astype(str) + '_' + profile_before['researchdate'].astype(str)
    ots_copy = ots_data.copy()
    ots_copy['key'] = ots_copy['SubjectID'].astype(str) + '_' + ots_copy['researchdate'].astype(str)
    ots_with_profile = ots_copy.merge(profile_before[['key', profile_col]], on='key', how='left')
    before = ots_with_profile.groupby(profile_col)['daily_ots'].sum()
    ots_clean = ots_with_profile[~ots_with_profile['key'].isin(anom_copy['key'])]
    after = ots_clean.groupby(profile_col)['daily_ots'].sum()
    pct_change = (after / before * 100).sort_values()
    if len(pct_change) == 0:
        return
    plt.figure(figsize=(12, max(6, len(pct_change) * 0.3)))
    bars = plt.barh(pct_change.index.astype(str), pct_change.values, color='teal')
    plt.xlabel('Процент сохранённого OTS после очистки (%)')
    plt.title(title)
    plt.axvline(x=100, color='red', linestyle='--', label='100% (без изменений)')
    for bar, pct in zip(bars, pct_change.values):
        if pct < 99.5:
            plt.text(pct + 0.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG['plots_dir'] / filename, dpi=150)
    plt.close()

# График изменения OTS по характеристикам ресурса
def plot_resource_ots_change(df_raw, ots_data, anomalies_df, resource_col, title, filename):
    if resource_col not in df_raw.columns:
        print(f"Колонка '{resource_col}' отсутствует, пропуск {filename}")
        return
    anom_copy = anomalies_df.copy()
    anom_copy['key'] = anom_copy['SubjectID'].astype(str) + '_' + anom_copy['researchdate'].astype(str)
    res_before = df_raw[['SubjectID', 'researchdate', resource_col]].drop_duplicates()
    res_before['key'] = res_before['SubjectID'].astype(str) + '_' + res_before['researchdate'].astype(str)
    ots_copy = ots_data.copy()
    ots_copy['key'] = ots_copy['SubjectID'].astype(str) + '_' + ots_copy['researchdate'].astype(str)
    ots_with_res = ots_copy.merge(res_before[['key', resource_col]], on='key', how='left')
    before = ots_with_res.groupby(resource_col)['daily_ots'].sum()
    ots_clean = ots_with_res[~ots_with_res['key'].isin(anom_copy['key'])]
    after = ots_clean.groupby(resource_col)['daily_ots'].sum()
    pct_change = (after / before * 100).sort_values()
    if len(pct_change) == 0:
        return
    plt.figure(figsize=(12, max(6, len(pct_change) * 0.3)))
    bars = plt.barh(pct_change.index.astype(str), pct_change.values, color='darkorange')
    plt.xlabel('Процент сохранённого OTS после очистки (%)')
    plt.title(title)
    plt.axvline(x=100, color='red', linestyle='--', label='100% (без изменений)')
    for bar, pct in zip(bars, pct_change.values):
        if pct < 99.5:
            plt.text(pct + 0.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG['plots_dir'] / filename, dpi=150)
    plt.close()

# Основной блок
if __name__ == "__main__":
    df_raw = load_all_data(CONFIG['data_path'])
    ots_data = prepare_ots_data(df_raw)
    print(f"Уникальных троек (SubjectID, дата, бренд): {len(ots_data)}")

    shares_data = compute_brand_shares(ots_data)
    print(f"Записей с долями: {len(shares_data)}")

    anomalies_df, reasons_df = run_anomaly_detection(shares_data)

    anomalies_df.to_csv(CONFIG['output_dir'] / 'anomalies.csv', index=False)
    reasons_df.to_csv(CONFIG['output_dir'] / 'anomaly_reasons.csv', index=False)
    print(f"Уникальных пар (SubjectID, researchdate) для удаления: {len(anomalies_df)}")
    print(f"Детальных записей в anomaly_reasons.csv: {len(reasons_df)}")

    # Статистика очистки
    total_subjects_before = df_raw['SubjectID'].nunique()
    total_anomalous_subjects = anomalies_df['SubjectID'].nunique()
    anom_keys = set(anomalies_df['SubjectID'].astype(str) + '_' + anomalies_df['researchdate'].astype(str))
    ots_keys = ots_data['SubjectID'].astype(str) + '_' + ots_data['researchdate'].astype(str)
    ots_after_total = ots_data[~ots_keys.isin(anom_keys)]['daily_ots'].sum()
    ots_before_total = ots_data['daily_ots'].sum()
    print(f"Статистика очистки")
    print(f"Уникальных респондентов до: {total_subjects_before}")
    print(f"Уникальных респондентов с аномалиями: {total_anomalous_subjects} ({total_anomalous_subjects/total_subjects_before*100:.2f}%)")
    print(f"Суммарный OTS до: {ots_before_total:.2f}")
    print(f"Суммарный OTS после: {ots_after_total:.2f} (сохранено {ots_after_total/ots_before_total*100:.2f}%)")

    # 1. Общий OTS по дням
    ots_before, ots_after = calculate_total_ots(ots_data, anomalies_df)
    plt.figure(figsize=(12, 5))
    plt.plot(ots_before.index, ots_before.values, marker='o', label='До удаления', linewidth=2)
    plt.plot(ots_after.index, ots_after.values, marker='s', label='После удаления', linewidth=2)
    plt.xlabel('Дата')
    plt.ylabel('Суммарный дневной OTS')
    plt.title('Изменение общего OTS после удаления аномальных респондентов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CONFIG['plots_dir'] / 'total_ots_before_after.png', dpi=150)
    plt.close()

    # 2. Изменение OTS по категориям CategoryDelivery
    cat_pct = calculate_category_ots_change(ots_data, anomalies_df)
    plt.figure(figsize=(12, max(6, len(cat_pct) * 0.3)))
    bars = plt.barh(cat_pct.index, cat_pct.values, color='steelblue')
    plt.xlabel('Процент сохранённого OTS после очистки (%)')
    plt.title('Изменение OTS по категориям CategoryDelivery')
    plt.axvline(x=100, color='red', linestyle='--', label='100% (без изменений)')
    for bar, pct in zip(bars, cat_pct.values):
        if pct < 99:
            plt.text(pct + 0.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center')
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONFIG['plots_dir'] / 'category_ots_change.png', dpi=150)
    plt.close()

    # 3. Количество аномальных респондентов по дням
    daily_anomaly_count = anomalies_df.groupby('researchdate').size().sort_index()
    plt.figure(figsize=(14, 5))
    plt.bar(daily_anomaly_count.index.astype(str), daily_anomaly_count.values, color='coral')
    dates = daily_anomaly_count.index.astype(str)
    n_dates = len(dates)
    step = max(1, n_dates // 15)
    plt.xticks(dates[::step], rotation=90, fontsize=8)
    plt.xlabel('Дата')
    plt.ylabel('Количество аномальных респондентов')
    plt.title('Количество аномальных респондентов по дням')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG['plots_dir'] / 'daily_anomaly_count.png', dpi=150)
    plt.close()

    # Дополнительные аналитические графики (п.8.2)
    plot_profile_ots_change(df_raw, ots_data, anomalies_df, 'Пол', 'Изменение OTS по полу', 'gender_ots_change.png')
    plot_profile_ots_change(df_raw, ots_data, anomalies_df, 'Возраст', 'Изменение OTS по возрасту', 'age_ots_change.png')
    plot_profile_ots_change(df_raw, ots_data, anomalies_df, 'Регион', 'Изменение OTS по региону', 'region_ots_change.png')
    plot_profile_ots_change(df_raw, ots_data, anomalies_df, 'Федеральный_округ', 'Изменение OTS по федеральному округу', 'federal_district_ots_change.png')
    plot_resource_ots_change(df_raw, ots_data, anomalies_df, 'ResourceName', 'Изменение OTS по названию ресурса', 'resourcename_ots_change.png')
    plot_resource_ots_change(df_raw, ots_data, anomalies_df, 'ResourceType', 'Изменение OTS по типу ресурса', 'resourcetype_ots_change.png')
    plot_resource_ots_change(df_raw, ots_data, anomalies_df, 'Platform', 'Изменение OTS по платформе', 'platform_ots_change.png')
    plot_resource_ots_change(df_raw, ots_data, anomalies_df, 'UseType', 'Изменение OTS по типу использования', 'usetype_ots_change.png')
    plot_profile_ots_change(df_raw, ots_data, anomalies_df, 'Category1', 'Изменение OTS по Category1', 'category1_ots_change.png')
    plot_profile_ots_change(df_raw, ots_data, anomalies_df, 'Category2', 'Изменение OTS по Category2', 'category2_ots_change.png')
    plot_profile_ots_change(df_raw, ots_data, anomalies_df, 'Category3', 'Изменение OTS по Category3', 'category3_ots_change.png')

    # Поисковые запросы первого аномального респондента
    if len(anomalies_df) > 0:
        sample_subject = anomalies_df.iloc[0]['SubjectID']
        sample_date = anomalies_df.iloc[0]['researchdate']
        cat_col_raw = 'CategoryNameDelivery' if 'CategoryNameDelivery' in df_raw.columns else 'CategoryDelivery'
        sample_queries = df_raw[
            (df_raw['SubjectID'] == sample_subject) &
            (df_raw['researchdate'] == sample_date)
        ][['QueryText', 'Brand', cat_col_raw]]
        print(f"\nПоисковые запросы аномального респондента {sample_subject} за {sample_date}:")
        print(sample_queries.to_string(index=False))
    else:
        print("Аномалий не найдено.")

    # График изменения OTS по дням для выбранного бренда
    if len(reasons_df) > 0:
        sample_brand_row = reasons_df.iloc[0]
        sample_brand_id = sample_brand_row['BrandID']
        sample_cat = sample_brand_row['CategoryDelivery']
        brand_ots_before = ots_data[
            (ots_data['BrandID'] == sample_brand_id) &
            (ots_data['CategoryDelivery'] == sample_cat)
        ].groupby('researchdate')['daily_ots'].sum()
        anom_copy = anomalies_df.copy()
        anom_copy['key'] = anom_copy['SubjectID'].astype(str) + '_' + anom_copy['researchdate'].astype(str)
        ots_copy = ots_data.copy()
        ots_copy['key'] = ots_copy['SubjectID'].astype(str) + '_' + ots_copy['researchdate'].astype(str)
        ots_clean = ots_copy[~ots_copy['key'].isin(anom_copy['key'])]
        brand_ots_after = ots_clean[
            (ots_clean['BrandID'] == sample_brand_id) &
            (ots_clean['CategoryDelivery'] == sample_cat)
        ].groupby('researchdate')['daily_ots'].sum()
        plt.figure(figsize=(12, 5))
        plt.plot(brand_ots_before.index, brand_ots_before.values, marker='o', label='До удаления')
        plt.plot(brand_ots_after.index, brand_ots_after.values, marker='s', label='После удаления')
        plt.xlabel('Дата')
        plt.ylabel('Суммарный дневной OTS бренда')
        plt.title(f'Изменение OTS бренда {sample_brand_row["Brand"]} ({sample_cat})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(CONFIG['plots_dir'] / 'brand_ots_before_after.png', dpi=150)
        plt.close()
    else:
        print("Аномалий не найдено, график для бренда не построен.")

    print("Решение выполнено.")
    print(f"Файлы сохранены в {CONFIG['output_dir']}")