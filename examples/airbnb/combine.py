import pandas as pd

users = pd.read_csv('train_users.csv')
countries = pd.read_csv('countries.csv')
age = pd.read_csv('age_gender_bkts.csv')
res = pd.merge(users, countries, on='country_destination')
res = pd.merge(res, age, on='country_destination')
res.to_csv('combined.csv', index=False)
