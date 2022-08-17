# Import Libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Importing movies.csv to python

df = pd.read_csv(r'C:\Users\Ar_Da\OneDrive\Desktop\SQL portfolio project\PythonProjects\movies.csv')

#Set that the list shown show all columns

pd.set_option('display.max_columns',None)

## DATA CLEANING
# Checking if there is a missing data and drop them

print(df.isnull().sum())

df=df.dropna()

#Checking data types for our columns

print(df.dtypes)

#Changing datatypes

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

#drop any duplicates

df=df.drop_duplicates()

##DATA ANALYSIS
#See if there is a correlation between budget and gross

plt.scatter(x=df['budget'], y=df['gross'], s=5)
plt.title('Gross Earning VS Budget')
plt.xlabel('Budget for Film')
plt.ylabel('Gross Earning')
sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color':'red', 's':5})
plt.show()

#Look at the correlation matrix
correlation_matrix  = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movies Features')
plt.ylabel('Movies Features')
plt.show()

#Look if company correlate with gross earning

df_numerized = df