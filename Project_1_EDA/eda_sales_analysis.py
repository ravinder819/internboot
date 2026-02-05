import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')

train = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
holidays = pd.read_csv('holidays_events.csv')
train.head()
train.info()
train.describe()

train['date'] = pd.to_datetime(train['date'])
train.isnull().sum()

daily_sales = train.groupby('date')['sales'].sum()

plt.figure(figsize=(12,5))
plt.plot(daily_sales)
plt.title("Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

train['month'] = train['date'].dt.month
monthly_sales = train.groupby('month')['sales'].mean()

plt.figure(figsize=(8,4))
monthly_sales.plot(kind='bar')
plt.title("Average Monthly Sales")
plt.show()

top_stores = train.groupby('store_nbr')['sales'].sum()
sort_values(ascending=False).head(10)

plt.figure(figsize=(8,4))
top_stores.plot(kind='bar')
plt.title("Top 10 Stores by Sales")
plt.show()

holidays['date'] = pd.to_datetime(holidays['date'])
holiday_sales = train.merge(holidays, on='date', how='left')
holiday_sales['is_holiday'] = holiday_sales['type'].notna()

holiday_sales.groupby('is_holiday')['sales'].mean()

holiday_sales[['date', 'type', 'sales']].head(10)
