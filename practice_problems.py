import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# Problem 1: Implement a simple neural network from scratch
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # Calculate gradients
        dZ2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 * (1 - self.a1))
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# Problem 2: Implement a data preprocessing pipeline
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = X.mean()
        self.std = X.std()
        return self
    
    def transform(self, X):
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Remove outliers
        X = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
        
        # Standardize features
        X = (X - self.mean) / self.std
        
        return X

# Problem 3: Implement a basic recommendation system
class SimpleRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
    
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        # Calculate cosine similarity
        self.similarity_matrix = np.dot(user_item_matrix, user_item_matrix.T)
        norms = np.array([np.sqrt(np.diagonal(self.similarity_matrix))])
        self.similarity_matrix = self.similarity_matrix / (norms * norms.T)
    
    def recommend(self, user_id, n_recommendations=5):
        user_ratings = self.user_item_matrix[user_id]
        similar_users = np.argsort(self.similarity_matrix[user_id])[::-1][1:]
        
        recommendations = []
        for item in range(self.user_item_matrix.shape[1]):
            if user_ratings[item] == 0:  # Item not rated by user
                score = 0
                total_similarity = 0
                for similar_user in similar_users:
                    if self.user_item_matrix[similar_user, item] > 0:
                        score += self.similarity_matrix[user_id, similar_user] * self.user_item_matrix[similar_user, item]
                        total_similarity += self.similarity_matrix[user_id, similar_user]
                if total_similarity > 0:
                    recommendations.append((item, score / total_similarity))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]

# Example usage
if __name__ == "__main__":
    # Example 1: Neural Network
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, (100, 1))
    
    nn = SimpleNeuralNetwork(5, 10, 1)
    for _ in range(1000):
        nn.forward(X)
        nn.backward(X, y, 0.01)
    
    # Example 2: Data Preprocessing
    df = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)
    processed_data = preprocessor.transform(df)
    
    # Example 3: Recommendation System
    user_item_matrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1]
    ])
    
    recommender = SimpleRecommender()
    recommender.fit(user_item_matrix)
    recommendations = recommender.recommend(0) 