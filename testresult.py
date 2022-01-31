import pandas as pd

df=pd.read_csv('evaluation/result.csv')
df=df.sort_values(by=['solving'])
print(df)
df=df[df.solving==df.iloc[-1].solving]
print(df)
df=df.sort_values(by=['time'])
print(df)
print('\nBest Solution:',df.iloc[0]['method'])
