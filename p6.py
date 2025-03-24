import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('user_course_data.csv')  
print(df.head()) 

#(eda 2)
course_popularity = df['course_category'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=course_popularity.index, y=course_popularity.values)
plt.title('Course Category Popularity')
plt.xlabel('Course Category')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.show()

print("Most Popular Course Category:", course_popularity.idxmax())
print("Least Popular Course Category:", course_popularity.idxmin())


#(eda 1)
age_bins = [0, 18, 25, 35, 45, 60, 100]  # Define age ranges
age_labels = ['0-18', '19-25', '26-35', '36-45', '46-60', '60+']
df['age_group'] = pd.cut(df['user_age'], bins=age_bins, labels=age_labels)

age_distribution = df['age_group'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(age_distribution.values, labels=age_distribution.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title('Age Group Distribution')
plt.show()

#(eda 3)
df['enrollment_date'] = pd.to_datetime(df['enrollment_date'], format='%d-%m-%Y')
df['week'] = df['enrollment_date'].dt.isocalendar().week
df['year'] = df['enrollment_date'].dt.year

weekly_trends = df.groupby(['year', 'week']).agg(
    total_time_spent=('time_spent_mins', 'sum'),
    total_logins=('number_of_logins', 'sum')
).reset_index()

weekly_trends['year_week'] = weekly_trends['year'].astype(str) + '-W' + weekly_trends['week'].astype(str)

plt.figure(figsize=(14, 7))
sns.lineplot(x='year_week', y='total_time_spent', data=weekly_trends, marker='o', label='Total Time Spent (mins)', color='blue')
plt.title('Weekly Trends in Total Time Spent')
plt.xlabel('Year-Week')
plt.ylabel('Total Time Spent (minutes)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='year_week', y='total_logins', data=weekly_trends, marker='o', label='Total Number of Logins', color='green')
plt.title('Weekly Trends in Total Number of Logins')
plt.xlabel('Year-Week')
plt.ylabel('Total Number of Logins')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
print(weekly_trends)


#(eda 4)
completion_by_category = df.groupby(['course_category', 'completion_status']).size().reset_index(name='count')

completion_pivot = completion_by_category.pivot(index='course_category', columns='completion_status', values='count').fillna(0)

completion_pivot['completion_rate'] = (completion_pivot.get('Completed', 0) / completion_pivot.sum(axis=1)) * 100

plt.figure(figsize=(12, 6))
sns.barplot(x=completion_pivot.index, y=completion_pivot['completion_rate'], palette='coolwarm')
plt.title('Completion Rates by Course Category')
plt.xlabel('Course Category')
plt.ylabel('Completion Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(completion_pivot[['completion_rate']])



#(Behavioral Analysis 1)
time_completion = df.groupby('completion_status')['time_spent_mins'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='completion_status', y='time_spent_mins', data=time_completion, palette='coolwarm')
plt.title('Average Time Spent by Completion Status')
plt.xlabel('Completion Status')
plt.ylabel('Average Time Spent (minutes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(time_completion)


#(Behavioral Analysis 2)
login_completion = df.groupby('completion_status')['number_of_logins'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='completion_status', y='number_of_logins', data=login_completion, palette='viridis')
plt.title('Average Number of Logins by Completion Status')
plt.xlabel('Completion Status')
plt.ylabel('Average Number of Logins')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(login_completion)


#(Behavioral Analysis 3)
completion_distribution = df['completion_status'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=completion_distribution.index, y=completion_distribution.values, palette='coolwarm')
plt.title('Distribution of Completion Status')
plt.xlabel('Completion Status')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(completion_distribution)
