import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the wine dataset
wine_df = pd.read_csv("wine/winequality-red.csv", sep=';')

# Separate features and target labels
wine_data = wine_df.iloc[:, :-1] # Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

wine_target = wine_df.iloc[:, -1] # Target label: quality (score between 0 and 10)

# Split data into training and testing sets (75% train, 25% test)
# Random state is taken as 42 because it is the answer to everything
X_train, X_test, y_train, y_test = train_test_split(wine_data, wine_target, test_size=0.25, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the Feed-Forward Neural Network (FFNN) model
ffnn = MLPClassifier(hidden_layer_sizes=(70,70), activation='relu', solver='adam', max_iter=1700)

# Train the model on the traning data
ffnn.fit(X_train, y_train)

# Make predictions on the testing data
predictions = ffnn.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions,zero_division=0))