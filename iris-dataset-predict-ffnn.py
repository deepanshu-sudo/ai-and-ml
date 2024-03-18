from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the iris dataset
iris = load_iris()
iris_data = iris.data # Features: sepal length, sepal width, petal length, petal width
iris_target = iris.target # Target labels: Iris Setosa (0), Versicolor (1), Virginica (2)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.2, random_state=42)

# Define the Feed-Forward Neural Network (FFNN) model
ffnn = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, activation='relu', solver='adam', random_state=42)

# Explanation of FFNN concepts:

# * Multilayer Perceptron (MLP): This is a type of artificial neural network
#   with an input layer, one or more hidden layers, and an output layer.

# * Hidden Layers: These layers are not directly connected to the input or
#   output and are responsible for learning complex relationships between
#   features. The number of hidden layers and neurons in each layer are
#   hyperparameters that can be tuned to improve model performance. Here, we
#   use three hidden layers with 10 neurons each.

# * Activation Function: This function introduces non-linearity into the
#   network's decision-making process. ReLU (Rectified Linear Unit) is a
#   popular choice that sets negative values to zero and allows positive
#   values to pass through unchanged.

# * Training: The model learns by adjusting the weights and biases between
#   neurons based on the training data. The optimizer (Adam in this case)
#   guides this process to minimize the error between predictions and actual
#   labels.

# * Maximum Iterations: This parameter limits the training time to prevent
#   overfitting, where the model performs well on training data but poorly
#   on unseen data. 

# Train the model on the training data
ffnn.fit(X_train, y_train)

# Make predictions on the testing data
predictions = ffnn.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))