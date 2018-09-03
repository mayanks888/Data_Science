import pandas as pd
import matplotlib
import seaborn

data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/House Prices: Advanced Regression Techniques/all(2)/train.csv')
print(data.head())

features=data.iloc[:,0:-1].values
label=data.iloc[:,-1].values
print(features[7])