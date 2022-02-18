import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

pd.set_option('display.float_format', lambda x: '%.4f' % x)


#####PREPROCESSING FOR REGRESSION DATASET######

# Creating New Column with Brand names
df["Brand"] = ""

for i in range(len(df)):
    df.at[i,'Brand'] = df.loc[i, "name"].split()[0]
    
    
# Filtering Values between Q1-1.5IQR and Q3+1.5IQR
def removeOutliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    data[data[col] > Q3] = Q3
    data[data[col] < Q1] = Q1

removeOutliers(df,'selling_price')
    

# Normalizing km_driven and selling_price
cols_to_norm = ['km_driven','selling_price']
df[cols_to_norm] = StandardScaler().fit_transform(df[cols_to_norm])

print(df.head())
