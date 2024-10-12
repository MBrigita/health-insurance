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

# Distribution of Numerical Features
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
sns.histplot(df['Age'], kde=True, bins=30, ax=axes[0])
axes[0].set_title('Distribution of Age')
sns.histplot(df['Annual_Premium'], kde=True, bins=30, ax=axes[1])
axes[1].set_title('Distribution of Annual Premium')
sns.histplot(df['Vintage'], kde=True, bins=30, ax=axes[2])
axes[2].set_title('Distribution of Vintage')
plt.tight_layout()
plt.show()

# Count Plots of Categorical Features
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
sns.countplot(x='Gender', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Count of Gender')
sns.countplot(x='Vehicle_Age', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Count of Vehicle Age')
sns.countplot(x='Vehicle_Damage', data=df, ax=axes[0, 2])
axes[0, 2].set_title('Count of Vehicle Damage')
sns.countplot(x='Previously_Insured', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Count of Previously Insured')
sns.countplot(x='Response', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Count of Response')
fig.delaxes(axes[1, 2])
plt.tight_layout()
plt.show()

# Box Plots to Compare Categories
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x='Response', y='Age', data=df, ax=axes[0])
axes[0].set_title('Age vs Response')
sns.boxplot(x='Response', y='Annual_Premium', data=df, ax=axes[1])
axes[1].set_title('Annual Premium vs Response')
plt.tight_layout()
plt.show()

# Pairplot for Multivariate Analysis
sns.pairplot(df[['Age', 'Annual_Premium', 'Vintage', 'Response']], hue='Response', height=2.5)
plt.show()

# Example: Count plot of Response based on Vehicle Age
plt.figure(figsize=(6, 4))
sns.countplot(x='Vehicle_Age', hue='Response', data=df)
plt.title('Response based on Vehicle Age')
plt.xlabel('Vehicle Age')
plt.ylabel('Count')
plt.legend(title='Response', loc='upper right', labels=['Not Interested', 'Interested'])
plt.show()

# Box Plots to Compare Categories betweeen the age of vechile and age of driver
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x='Vehicle_Age', y='Age', data=df, ax=axes[0])
axes[0].set_title('Age vs Vechile Age')
sns.boxplot( x='Response',y='Age', data=df, ax=axes[1])
axes[1].set_title(' Response vs Age')
plt.tight_layout()
plt.show()

#to identify correlated variables we do correaltion matrix,
# Correlation is as a mean to see which variables are important as well as to investigate any multicollinearity between our independent predictors.
correlation = df[numerical_columns + ['Response']].corr()
correlation

# Set up the matplotlib figure
plt.figure(figsize=(5, 4))

# Create a heatmap with the correlation matrix
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)

# Set plot title and labels
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.show()