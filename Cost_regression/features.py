import pandas as pd
import numpy as np
import joblib
import json
import ast

def EDA (df):
    # df = df.drop_duplicates()
    df = df.replace('[]', np.nan)
    df['published_at'] = pd.to_datetime(df['published_at']).dt.to_period("Y")
    with open("artifacts/list_to_drop.json", "r") as f:
        lst_to_drop = json.load(f)
    df.drop(columns=[c for c in lst_to_drop if c in df.columns], inplace=True)
    return df


def location(df):
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
    
    df['area'] = df['area'].apply(extract_city)
    city_counts = df['area'].value_counts()
    with open("artifacts/city_area_dict.json", "r") as f:
        city_dict = json.load(f)
    df['city'] = df['area'].apply(lambda x: city_dict[x] if x in city_dict else categorize_city(x, city_counts))
    df.drop(['area', 'region'], axis=1, inplace=True, errors="ignore")

    df.drop('address', axis=1, inplace=True, errors="ignore")

    with open("artifacts/city_mean_salary.json", "r") as f:
        city_mean_salary_dict = json.load(f)
    df['city_mean_salary'] = df['city'].map(city_mean_salary_dict)
    df.drop('city', axis=1, inplace=True, errors="ignore")
    return df

def json_cols(df):
    df['type'] = df['type'].apply(lambda x: json.loads(x.replace("'", '"')).get('name') if pd.notna(x) else None)
    df['is_open'] = df['type'].replace({"Открытая" : 1, "Анонимная" : 0, "Рекламная" : 1}).astype(int)
    df.drop('type', axis = 1, inplace = True)

    df['schedule'] = df['schedule'].apply(lambda x: json.loads(x.replace("'", '"')).get('name') if pd.notna(x) else None)
    
    df['experience'] = df['experience'].apply(lambda x: json.loads(x.replace("'", '"')).get('name') if pd.notna(x) else None)
    df['experience'].replace({"Нет опыта" : 0, "От 1 года до 3 лет" : 1, "От 3 до 6 лет" : 2, "Более 6 лет" : 3}, inplace=True)

    return df

def name_with_emb(df):
    model = joblib.load("models/emb_model.pkl")
    embeddings_name = model.encode(df['name'].str.lower().replace(r'[\W_]+', ' ', regex=True).tolist()) # Приводим к нижнему регистру, заменяем небуквенно-цифровые символы на пробелы, преобразуем в список строк
    emb_df = pd.DataFrame(embeddings_name, columns=[f'name_emb_{i}' for i in range(embeddings_name.shape[1])])
    df = pd.concat([df, emb_df], axis=1)
    return df

def extr_key_skills(df):
    def extract_key_skills_from_list(key_skills_list):
        if pd.isna(key_skills_list):
            return np.nan
        else:
            return [d['name'] for d in ast.literal_eval(key_skills_list)]
    df['key_skills'] = df['key_skills'].apply(lambda x: extract_key_skills_from_list(x))
    
    df['is_developer'] = df['name'].str.contains('программ|разраб|developer|dev', case=False)
    df['is_data_analyst'] = df['name'].str.contains('data|scientist|analytics|аналитик|анализ|данн', case=False)
    df['is_rieltor'] = df['name'].str.contains('риелтор|недвижимост|продаж', case=False)
    df['is_lead'] = df['name'].str.contains('начальник|lead', case=False)
    df['is_senior'] = df['name'].str.contains('ведущий|руководитель|senior|старший', case=False)

    df.drop(['key_skills', 'name'], axis=1, inplace=True)

    return df

def schedule_one_hot(df):

    # Для schedule
    schedule_dummies = pd.get_dummies(df['schedule'], prefix='schedule')
    # schedule_cats = ['schedule_' + str(x) for x in df['schedule'].unique()]
    # if schedule_cats:
    #     # Добавляем отсутствующие колонки
    #     for col_name in schedule_cats:
    #         if col_name not in schedule_dummies.columns:
    #             schedule_dummies[col_name] = 0
    #     # # Упорядочиваем колонки
    #     # schedule_dummies = schedule_dummies.reindex(columns=[f'schedule_{cat}' for cat in schedule_cats], fill_value=0)
    
    df = pd.concat([df, schedule_dummies], axis=1)
    df.drop('schedule', axis=1, inplace=True)
    return df

def prepare_features(X):
    X = EDA(X)
    X = location(X)
    X = json_cols(X)
    X = name_with_emb(X)
    X = extr_key_skills(X)
    X = schedule_one_hot(X)
    return X