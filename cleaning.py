import pandas as pd
import numpy as np


df=pd.read_csv('dataset.csv')
df_final=df[['Month','DayofMonth','DayOfWeek','ArrDelay','DepDelay','Origin','Dest','Distance','CarrierDelay','NASDelay','WeatherDelay','LateAircraftDelay','SecurityDelay']]

print(df_final.isnull().values.any())
print(df_final.isnull().sum())

df_final.to_csv('flights.csv') 