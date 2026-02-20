import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
# Load dataset
df = pd.read_csv("german.csv", sep=';')

# Check target values
print("Target Distribution:")
print(df['Creditability'].value_counts())

df['Creditability'] = df['Creditability'].astype(int)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 1 Default Rate
default_rate = df['Creditability'].mean()
print("\nOverall Default Rate:", round(default_rate*100,2), "%")

# 2 Average Loan Amount
print("\nAverage Credit Amount by Risk:")
print(df.groupby('Creditability')['Credit_Amount'].mean())

# 3 Average Loan Duration
print("\nAverage Duration by Risk:")
print(df.groupby('Creditability')['Duration_of_Credit_monthly'].mean())

# 4 High Loan + Long Duration Risk
high_risk_segment = df[
    (df['Credit_Amount'] > df['Credit_Amount'].median()) &
    (df['Duration_of_Credit_monthly'] > df['Duration_of_Credit_monthly'].median())
]

segment_default_rate = high_risk_segment['Creditability'].mean()
print("\nHigh Loan + Long Duration Default Rate:",
      round(segment_default_rate*100,2), "%")

# 5 Age-Based Risk Segmentation
df['AgeGroup'] = pd.cut(df['Age_years'], bins=[18,25,35,50,100])

print("\nDefault Rate by Age Group:")
print(df.groupby('AgeGroup')['Creditability'].mean())

# 6 Feature Importance (Random Forest)
X = df.drop(['Creditability','AgeGroup'], axis=1)
y = df['Creditability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nTop 5 Important Features (Random Forest):")
print(importance.sort_values(ascending=False).head(5))

# 7 Logistic Regression Coefficients
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)

coefficients = pd.Series(log.coef_[0], index=X.columns)
print("\nTop 5 Positive Risk Factors (Logistic Regression):")
print(coefficients.sort_values(ascending=False).head(5))

# 8 Risk Probability Scoring
df['RiskProbability'] = rf.predict_proba(X)[:,1]

print("\nRisk Probability Summary:")
print(df[['RiskProbability']].describe())

# 9 High-Risk Customer Percentage
high_risk = df[df['RiskProbability'] > 0.6]
print("\nHigh Risk Customers %:",
      round(len(high_risk)/len(df)*100,2), "%")

# 10 Rule-Based Risk Pattern
rule_based = df[
    (df['Duration_of_Credit_monthly'] > 36) &
    (df['Credit_Amount'] > 5000)
]

rule_default_rate = rule_based['Creditability'].mean()
print("\nRule-Based Segment Default Rate:",
      round(rule_default_rate*100,2), "%")

# 11 Model Evaluation
y_pred = rf.predict(X_test)

print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred)*100,2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))