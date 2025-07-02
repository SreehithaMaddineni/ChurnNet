# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
#df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")



# Drop customerID as it's not useful
df.drop('customerID', axis=1, inplace=True)

# Check datatypes
df.dtypes

# Remove rows where TotalCharges is blank
df = df[df.TotalCharges != ' ']

# Try to convert TotalCharges to float (expecting a warning)
try:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
except Exception as e:
    print("Error converting TotalCharges:", e)

# Quick check
df.TotalCharges.values[:5]

# Visualize tenure vs churn
plt.figure(figsize=(7,4))
plt.hist([df[df.Churn=='No'].tenure, df[df.Churn=='Yes'].tenure],
         color=['blue','orange'], label=['No','Yes'], rwidth=0.9)
plt.xlabel("Tenure")
plt.ylabel("Customers")
plt.title("Tenure distribution by Churn")
plt.legend()
plt.show()

# Visualize monthly charges vs churn
plt.figure(figsize=(7,4))
plt.hist([df[df.Churn=='No'].MonthlyCharges, df[df.Churn=='Yes'].MonthlyCharges],
         color=['blue','orange'], label=['No','Yes'], rwidth=0.9)
plt.xlabel("Monthly Charges")
plt.ylabel("Customers")
plt.title("Monthly Charges by Churn")
plt.legend()
plt.show()

# See unique values for object columns
def show_uniques(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            print(col, data[col].unique())

show_uniques(df)

# Replace 'No internet service' and 'No phone service' with 'No'
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# Convert Yes/No columns to 1/0
yn_cols = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity',
           'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
           'StreamingMovies','PaperlessBilling','Churn']
for col in yn_cols:
    df[col].replace({'Yes':1, 'No':0}, inplace=True)

# Convert gender to 1/0
df['gender'].replace({'Female':1, 'Male':0}, inplace=True)

# One-hot encode categorical columns
df2 = pd.get_dummies(df, columns=['InternetService','Contract','PaymentMethod'])

# Scale numerical columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df2[scale_cols] = scaler.fit_transform(df2[scale_cols])

# Prepare features and target
X = df2.drop('Churn', axis=1)
y = df2['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the ANN
import tensorflow as tf
from tensorflow import keras

# (Tried Adam, but sticking with SGD as per assignment)
# model.compile(optimizer='adam', ...)
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=50)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.2f}")

# Predict
y_prob = model.predict(X_test)
y_pred = [1 if p > 0.5 else 0 for p in y_prob]

# Classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))

# Confusion matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# (Mistake: tried to use a column that doesn't exist)
try:
    print(df2['NonExistentColumn'].value_counts())
except Exception as e:
    print("Column error:", e)
