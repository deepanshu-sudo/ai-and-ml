import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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
ffnn = MLPRegressor(hidden_layer_sizes=(100,50,25), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='adaptive' , learning_rate_init=0.007, max_iter=2000, random_state=42, early_stopping=True)

# Train the model on the traning data
ffnn.fit(X_train, y_train)

# Make predictions on the testing data
predictions = ffnn.predict(X_test)

# Evaluate the model's performance
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R-squared:", r2_score(y_test, predictions)) 