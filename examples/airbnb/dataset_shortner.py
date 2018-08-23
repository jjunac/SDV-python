import pandas as pd

df = pd.read_csv('combined.csv')
for n in [320000]:
    df.sample(n=n).to_csv('combined_%s.csv' % n, index=False)
