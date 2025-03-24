import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
df = pd.read_csv("Airline_Delay_Cause.csv")
print(df.head())
print(df.info())

airport_delays = df.groupby('airport_name')['arr_delay'].sum().sort_values(ascending=False)
print("Airports with the highest delays:")
print(airport_delays.head())


airline_delays = df.groupby('carrier_name')['arr_delay'].sum().sort_values(ascending=False)
print("Airlines with the highest delays:")
print(airline_delays.head())


weather_delays = df.groupby('airport_name')['weather_delay'].sum().sort_values(ascending=False)
print("Airports with the highest weather delays:")
print(weather_delays.head())
plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
sns.barplot(x=airport_delays.head().values, y=airport_delays.head().index)
plt.title('Top 5 Airports with Highest Delays')
plt.xlabel('Total Arrival Delay (minutes)')
plt.ylabel('Airport')

plt.subplot(1, 2, 2)
sns.barplot(x=airline_delays.head().values, y=airline_delays.head().index)
plt.title('Top 5 Airlines with Highest Delays')
plt.xlabel('Total Arrival Delay (minutes)')
plt.ylabel('Airline')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 8))
sns.barplot(x=weather_delays.head().values, y=weather_delays.head().index)
plt.title('Top 5 Airports with Highest Weather Delays')
plt.xlabel('Total Weather Delay (minutes)')
plt.ylabel('Airport')
plt.tight_layout()
plt.show()


# Histogram
plt.figure(figsize=(12, 6))
sns.histplot(df['arr_delay'], kde=True)
plt.title('Histogram of Arrival Delay')
plt.xlabel('Arrival Delay')
plt.ylabel('Frequency')
plt.show()

# Box Plot
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['arr_delay'])
plt.title('Box Plot of Arrival Delay')
plt.xlabel('Arrival Delay')
plt.show()

# Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(x=df['arr_delay'])
plt.title('Violin Plot of Arrival Delay')
plt.xlabel('Arrival Delay')
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Airline_Delay_Cause.csv')
weather_columns = ['weather_delay', 'weather_ct']  

# Correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(df[weather_columns + ['arr_delay']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between Weather Variables and Arrival Delay')
plt.show()

for col in weather_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=col, y='arr_delay')
    plt.title(f'Scatter Plot of {col} vs. Arrival Delay')
    plt.xlabel(col)
    plt.ylabel('Arrival Delay')
    plt.show()

for col in weather_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=col, y='arr_delay')
    plt.title(f'Box Plot of {col} vs. Arrival Delay')
    plt.xlabel(col)
    plt.ylabel('Arrival Delay')
    plt.show()
