import pandas as pd


# Load dataset
data = pd.read_csv('C:\cw2-int\data.csv')

# View the first few rows
print(data.head())

#Display dataset info
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Check for missing values
print(data.info())
print("missing values per col: \n", data.isnull().sum())
data.fillna(method='ffill', inplace=True)  # Fill missing values with forward fill

# Check for missing values 
print(data.isnull().sum()) 
# Optionally fill missing values 
data.fillna(method='ffill', inplace=True)

from sklearn.model_selection import train_test_split 
X = data.drop('class', axis=1)  # Drop target variable 
y = data['class']  # The target variable 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Normalize the feature values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score , classification_report
model = RandomForestClassifier() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("classification report:\n", classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
print(y.value_counts())


from sklearn.linear_model import LogisticRegression
simple_model = LogisticRegression(max_iter=1000)
simple_model.fit(X_train, y_train)
y_simple_pred = simple_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_simple_pred))




