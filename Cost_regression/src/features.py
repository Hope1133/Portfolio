def data_preparation(X):
    X = EDA(X, lst_to_drop)
    X = location(X, mean_salary_dict, city_dict)
    X = json_cols(X)
    X = name_with_emb(X)
    X = extr_key_skills(X)
    X = schedule_one_hot(X, schedule_categories)
    return X