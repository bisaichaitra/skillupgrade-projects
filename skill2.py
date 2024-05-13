import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the training and testing datasets
train_df = pd.read_csv(r'C:\Users\bisai\Downloads\titanic\train.csv')
test_df = pd.read_csv(r'C:\Users\bisai\Downloads\titanic\test.csv')

# Step 2: Preprocess the data
# Combine train and test datasets for preprocessing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Fill missing values for Age and Fare with median
combined_df['Age'] = combined_df['Age'].fillna(combined_df['Age'].median())
combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())

# Encode Sex feature (0 for male, 1 for female)
combined_df['Sex'] = combined_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Pclass feature
pclass_dummies = pd.get_dummies(combined_df['Pclass'], prefix='Pclass')
combined_df = pd.concat([combined_df, pclass_dummies], axis=1)

# Drop unnecessary columns
combined_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Pclass'], axis=1, inplace=True)

# Split the combined dataset back into train and test datasets
train_processed = combined_df[:len(train_df)]
test_processed = combined_df[len(train_df):]

# Step 3: Split the training dataset into features (X_train) and target (y_train)
X_train = train_processed.drop('Survived', axis=1)
y_train = train_processed['Survived']

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_processed.drop('Survived', axis=1))

# Step 5: Model Training (using Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions on the testing dataset
predictions = model.predict(X_test_scaled)

# Step 7: (Optional) Submit predictions to Kaggle competition
# submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
# submission_df.to_csv('submission.csv', index=False)
# Step 9: Evaluate the model's performance
# Make predictions on the training set
train_predictions = model.predict(X_train_scaled)

# Calculate accuracy on the training set
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Print the predictions
print(predictions)
