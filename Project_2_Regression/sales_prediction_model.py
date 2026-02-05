import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

train = pd.read_csv('train.csv')
holidays = pd.read_csv('holidays_events.csv')

train['date'] = pd.to_datetime(train['date'])
holidays['date'] = pd.to_datetime(holidays['date'])

train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['weekday'] = train['date'].dt.weekday

train = train.merge(
    holidays[['date', 'type']], 
    on='date', 
    how='left'
)

train['is_holiday'] = train['type'].notna().astype(int)

train.drop(columns=['type'], inplace=True)

train['onpromotion'] = train['onpromotion'].fillna(0)

features = [
    'store_nbr',
    'onpromotion',
    'year',
    'month',
    'day',
    'weekday',
    'is_holiday'
]

X = train[features]
y = train['sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

feature_importance = pd.Series(
    model.coef_,
    index=features
).sort_values(ascending=False)

feature_importance

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


