import pandas as pd
import numpy as np 
import seaborn as sns

#import data
df = pd.read_csv("train.csv")
#exploring data
df.head(10)

# separate target from predictors
df.dropna(axis=0, subset=['Response'], inplace=True)
y =df[['Response']]
df=df.set_index("id")

Missing_Percentage = (df.isnull().sum()).sum()/np.product(df.shape)*100
print("The number of missing entries before cleaning: " + str(round(Missing_Percentage,5)) + " %")

#split data to numeric and categorical
numerical_columns=['Age', 'Region_Code','Annual_Premium','Vintage']
categorical_columns=['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']

#checking the numerical columns
df.describe()