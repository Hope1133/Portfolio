import pandas as pd
import json

def columns_to_drop(df):
    missing = df.isna().mean().sort_values(ascending=False)
    lst_to_drop = list(missing[missing > 0.90].index)
    url_cols = [x for x in df.columns if 'url' in x]
    lst_to_drop = lst_to_drop + url_cols
    lst_to_drop.append('contacts')
    lst_to_drop = lst_to_drop + ['created_at', 'published_at']
    lst_to_drop = lst_to_drop + ['employer', 'snippet', 'description', 'specializations']
    with open("artifacts/list_to_drop.json", "w") as f:
        json.dump(lst_to_drop, f)

def city_area_dict(df):
    city_dict = dict(df[['area', 'city']].values)
    with open("artifacts/city_area_dict.json", "w") as f:
        json.dump(city_dict, f)

def calc_city_mean_salary(df):
    city_mean_salary_dict = df.groupby('city')['mean_salary'].mean().to_dict()
    with open("artifacts/city_mean_salary.json", "w") as f:
        json.dump(city_mean_salary_dict, f)

    print("City mean salary dict saved.")    
