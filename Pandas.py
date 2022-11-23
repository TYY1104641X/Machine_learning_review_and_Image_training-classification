# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:42:29 2022

@author: Yuanyuan_Tang

"""

from io import StringIO
import numpy as np
import pandas as pd


'''
Import data into Data Frame,  read_csv, 
DataFrame: in the structure of Dictionary

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
'''
#df0=pd.read_csv(data_csv)  # Read data infer the header based on the first row
# The following command added headers mannually
df1=pd.read_csv("Movie.csv", header=0, \
                names=['Rank','MovieTitle','Studio','TotalGross','Theaters','OpeningGross','OpeningTheaters','OpenDate'])  # Read data
#df2=pd.DataFrame(data_csv,
#\ columns=['Rank','MovieTitle','Studio','TotalGross','Theaters','OpeningGross','OpeningTheaters','OpenDate'])  # Read data

#print(df1.head(n=5))  # Show the first 5 rows of data
print(df1.tail(n=5))  # Show the last 5 rows of data


print(df1.head(n=19))

print(df1.columns[3])  # Read the name of Pandas DataFrame

print('------------------------------Read the Pandas-----------------------------')

'''
Write data in DataFrame to a csv
https://stackoverflow.com/questions/16923281/writing-a-pandas-dataframe-to-csv-file
'''
df1.to_csv('Movie.csv')


'''
Read specific columns, rows, and location
df.iloc[]--based on integer index
df.loc[]--based on label index
df['']
Read a specific column  df1['Rank']
https://www.educative.io/blog/pandas-cheat-sheet

https://www.statology.org/pandas-select-rows-by-index/
'''
a=df1['Rank']   # Read a specific column
b=df1['Rank'][:5]  # Get the first 5 elements of the first column

c=df1.iloc[2:5, -1] # Read the elements from index 2 to 5-1 of the last column
e=df1.iloc[[2,5,8], 0] # Read 2th,5th, 8th elements of first column
d=df1.iloc[0]    # 0-th row
f=df1.iloc[:,0]   # 0th column



'''
Retrieval information of pandas
read the size of Pandas DataFrame
Data types of DataFrame
https://pbpython.com/pandas_dtypes.html
'''
print(df1.dtypes)
df_shape=df1.shape  # The size of DataFrame
print(df1.info())  # See the datainformation of DataFrame
print('type=',type(df1.iloc[1,1]))

print('..........................................................')

'''
A toy example to change data type of strings
https://towardsdatascience.com/how-to-change-column-type-in-pandas-dataframes-d2a5548888f8
'''
df = pd.DataFrame(
  [
    ('1', 1, 'hi'),
    ('2', 2, 'bye'),
    ('3', 3, 'hello'),
    ('4', 4, 'goodbye'),
  ],
  columns=list('ABC')
)
print(df)
df.info()

# Change data type of the columns
df['A'] = df['A'].astype(int)
df.info()
df['B'] = df['B'].astype(str)
df.info()
df['B'] = df['B'].astype(float)
df.info()


'''
Pandas to numpy 
https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
'''
g=df[['A','B']].to_numpy()
print(type(g))

'''
Pandas DataFrame remove characters
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.replace.html
'''
import re
df2=df1.head(n=5)
Y=df2['TotalGross'].values
#print(Y)
for i in range(len(Y)):
    Y[i]=float(re.sub('[$,]','', Y[i])) # replace %, in [] by '' and convert to float
#print(Y)
df2['TotalGross']=Y
df2['TotalGross']=df2['TotalGross'].astype(float)

#print(df2.info())




'''
Pandas add a new column to specific positions
https://stackoverflow.com/questions/29517072/add-column-to-dataframe-with-constant-value
'''
Z=[10*i for i in range(1,5)]
df.insert(0,'Z',Z)   #Insert Z with name 'Z' to a specific column
#print(df)


'''
Drop a column or a row
https://www.analyticsvidhya.com/blog/2021/11/a-simple-guide-to-pandas-dataframe-operations/
'''
df3=df.drop(  labels=['C'],
axis=1,       # 1-drop column  
inplace=False  #creat a copy, otherwise derectly on df
)
#print(df3)

df4=df.drop(  labels=[1],
axis=0,       # 0-drop a row 
inplace=False  #creat a copy, otherwise derectly on df
)
#print(df4)


'''
Handle missing values NAN
https://www.analyticsvidhya.com/blog/2021/11/a-simple-guide-to-pandas-dataframe-operations/

Fill NA:
If you want to replace all the NA values with a specific number or especially with zero 
then you can use the fillna() function which will take various arguments.

DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)

value: Specific value with which NA will be replaced.
method: {“backfill”,”bfill”,”pad”,”ffill”,None}, these all are the methods to fill the na values.
If you want to replace the NA values from the backward
 and forward values present in the data then you can use “bfill” and “ffill”. 
axis: {0 or “index”,1 or”column”}, Axis along which to fill missing values.
limit: This is the maximum number of NA values to be filled with in the axis.
Let’s implement all of these methods:-
'''
# Generate a DataFrame with missing data
df5 = pd.DataFrame([[np.nan, 2, np.nan, 0], 
     [3, 4, np.nan, 1], 
     [np.nan, np.nan, np.nan, 5], 
     [np.nan, 3, np.nan, 4]], 
     columns=list("ABCD"))
print(df5)

# find NA values
print(df5.isna().sum())  # Count NANs for all columns
print(df5["D"].isna().sum()) # Count NANs for a specific column
print(df5.iloc[0].isna().sum()) # Count NANs for a specific row

# Fill NA with specific values by fllna
df6=df5["A"].fillna(100)  # Fill a specific column
print(df6)
df5['A']=df6   # Update a specific column
print(df5)
df6=df5.iloc[1].fillna(20)  # Fill a specific row
print(df6)


'''
Insert a date_list with period "D"
https://stackoverflow.com/questions/40858880/add-a-date-column-in-pandas-df-using-constant-value-in-str


 pandas.date_range: Add Time in a Range:
https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
'''

# Add a constant time stamp
df5['dates'] = pd.Timestamp('2016-11-06')
print(df5)
# Add a time stamp with specific period 
df5['dates']=pd.date_range(start='1/1/2018', periods=len(df5.iloc[:,0]), freq='D')
print(df5)


'''
Group by function: group data based on specific categorials an
https://stackoverflow.com/questions/63357396/calculate-mean-on-multiple-groups

https://www.analyticsvidhya.com/blog/2021/11/a-simple-guide-to-pandas-dataframe-operations/

'''

# import the pandas library
#import pandas as pd
points_table = {'Team_':['MI', 'CSK', 'Devils', 'MI', 'CSK',
   'RCB', 'CSK', 'CSK', 'KKR', 'KKR', 'KKR', 'RCB'],
   'Rank_' :[1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
   'Year_' :[2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
   'Point_':[876,789,863,673,741,812,756,788,694,701,804,690]}
df6= pd.DataFrame(points_table)
print(df6)

#Now group by the data according to the year
groupby_= df6.groupby('Year_')
#print(groupby_)
for team,group in groupby_:
   print(team)
   print(group)


g_m=df6.groupby('Year_').mean()
print(g_m)
print(g_m.loc[2015][1])  # Read the data in the ith row and j-th column

# Read the max-value
print(g_m['Point_'].idxmax()) # The index of max
print(g_m['Point_'].max()) # The index of max






