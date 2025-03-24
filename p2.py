import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Airline_Delay_Cause.csv")

print(df.isna().sum())

df.dropna(inplace=True)
print(df.info())

features = ['weather_delay', 'carrier_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
X = df[features]
y = df['arr_delay']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(coefficients)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test - y_pred)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Actual Arrival Delay')
plt.xlabel('Actual Arrival Delay')
plt.ylabel('Residuals')
plt.show()

feature_importance = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.index, y=feature_importance['Coefficient'])
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Predicted vs Actual Arrival Delay')
plt.xlabel('Actual Arrival Delay')
plt.ylabel('Predicted Arrival Delay')
plt.show()

