# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load the dataset
df = pd.read_csv('heart_disease_dataset.csv')

# Fill missing values with median for numerical features
numeric_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Convert target labels to binary (0 or 1)
# Assuming 'num' is the target column indicating heart disease
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Define the features and the target
X = df.drop(columns=['id', 'num'])  # Drop 'id' and 'num' columns
y = df['num']

# Define categorical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Create preprocessing pipelines for numerical and categorical data
numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_columns)
    ])

# Create the logistic regression model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=100))
])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Test Accuracy: {accuracy:.2%}')
print(f'Test F1-Score: {f1:.2%}')

# Save the model
joblib.dump(model_pipeline, 'heart_disease_model.pkl')
