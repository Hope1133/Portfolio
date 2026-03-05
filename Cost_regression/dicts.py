import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def columns_to_drop(df):
    missing = df.isna().mean().sort_values(ascending=False)
    lst_to_drop = list(missing[missing > 0.90].index)
    url_cols = [x for x in df.columns if 'url' in x]
    lst_to_drop = lst_to_drop + url_cols
    lst_to_drop.append('contacts')
    lst_to_drop = lst_to_drop + ['created_at', 'published_at']
    lst_to_drop = lst_to_drop + ['employer', 'snippet', 'description', 'specializations']
    lst_to_drop = lst_to_drop + ['relations', 'working_days', 'working_time_intervals', 'working_time_modes']
    with open(ARTIFACTS_DIR / "list_to_drop.json", "w") as f:
        json.dump(lst_to_drop, f)
        print("lst_to_drop was written")

def extract_city(area_str):
    try:
        area_dict = json.loads(area_str.replace("'", '"'))  # заменяем одинарные кавычки на двойные для корректного JSON
        return area_dict.get('name')
    except:
        return None
    
def categorize_city(city, city_counts):
    if city_counts.get(city, 0) > 1000:  # Крупные города
        return city
    elif city_counts.get(city, 0) > 100:   # Средние города
        return 'Средний город'
    else:                                  # Малые города
        return 'Маленький город'


def city_area_dict(df):
    df['area'] = df['area'].apply(extract_city)
    city_counts = df['area'].value_counts()
    df['city'] = df['area'].apply(lambda x: categorize_city(x, city_counts))
    city_dict = dict(df[['area', 'city']].values)
    with open(ARTIFACTS_DIR / "city_area_dict.json", "w") as f:
        json.dump(city_dict, f)

def calc_city_mean_salary(df):
    df['area'] = df['area'].apply(extract_city)
    city_counts = df['area'].value_counts()
    df['city'] = df['area'].apply(lambda x: categorize_city(x, city_counts))
    city_mean_salary_dict = dict(df.groupby('city')['mean_salary'].mean('mean_salary'))
    with open(ARTIFACTS_DIR / "city_mean_salary.json", "w") as f:
        json.dump(city_mean_salary_dict, f)
    print("City mean salary dict saved.")    
