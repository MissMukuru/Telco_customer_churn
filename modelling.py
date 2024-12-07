import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import joblib


def encode_categorical_features(df):
    """Encoding the categorical features of the data"""
    cat_features = df.select_dtypes(include='object')
    cat_features_encoded = pd.get_dummies(cat_features, drop_first=True)  # Use one-hot encoding for categorical features
    
    # Replace the original categorical columns with the encoded ones
    df = df.drop(columns=cat_features.columns)  # Drop the original categorical columns
    df = pd.concat([df, cat_features_encoded], axis=1)  # Concatenate the encoded columns back to the DataFrame

    return df

def define_features_target_variables(df):
    """Defining the feature and target variables"""
    X = df.drop('churn')  # Replace 'Churn' as the target variable
    y = df['churn']  # 'Churn' is a binary column: 1 for churned, 0 for not churned
    return X, y

def split_data(X, y):
    """Splitting the data"""
    return train_test_split(X, y, random_state=42, test_size=0.3)  # random_state = 42 ensures reproducibility

def scale_data(X_train, X_test):
    """Fitting the scaler onto the training data and applying the transformations to the test set"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Apply transformation to test set only
    return X_train_scaled, X_test_scaled, scaler

def apply_pca(X_train_scaled, X_test_scaled, n_components=2):
    """Applying PCA for dimensionality reduction"""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca, pca

def initialize_model():
    """Creating the model instances"""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42)
    }
    return models

def train_and_predict(models, X_train, X_test, y_train):
    """Train each model and make predictions"""
    prediction = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prediction[name] = y_pred
        
    return prediction

def evaluate_model_performance(models, prediction, y_test):
    """Evaluate each model's performance and return the best model"""
    metrics = []
    best_accuracy = 0
    best_model = None
    
    for name, y_pred in prediction.items():
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f'{name} - Accuracy: {accuracy}\n')
        print(f'Confusion Matrix:\n{cm}\n')
        print(f'Classification Report:\n{report}\n')
        
        metrics.append({
            'Name': name,
            'Accuracy': accuracy
        })
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = models[name]
            
    metrics_df = pd.DataFrame(metrics)
    
    return best_model, metrics_df

def visualize_decision_tree(model, feature_names, output_filename='decision_tree.png'):
    """Visualize and save the decision tree if the model is a DecisionTreeClassifier"""
    if isinstance(model, DecisionTreeClassifier):
        plt.figure(figsize=(20, 10))
        plot_tree(model, feature_names=feature_names, filled=True, rounded=True, class_names=["Not Churned", "Churned"])
        
        # Save the plot as a file (e.g., PNG)
        plt.savefig(output_filename, format='png')
        plt.close()  # Close the plot to avoid display issues when running the script
        print(f"Decision tree saved as {output_filename}")

def main():
    # Loading the data
    df = pd.read_csv('Telco_Customer_Churn.csv')  # Load the churn dataset
    
    # Encoding the categorical features
    encoded_df = encode_categorical_features(df)
    
    # Define features and target variables
    X, y = define_features_target_variables(encoded_df)
    
    # Splitting the data into test and train sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Apply PCA for dimensionality reduction
    X_train_pca, X_test_pca, pca = apply_pca(X_train_scaled, X_test_scaled, n_components=2)

    # Create model instance
    models = initialize_model()

    # Train and predict
    predictions = train_and_predict(models, X_train_pca, X_test_pca, y_train)

    # Evaluate the model
    best_model, metrics_df = evaluate_model_performance(models, predictions, y_test)

    # Visualize the decision tree if it's the best model
    if isinstance(best_model, DecisionTreeClassifier):
        visualize_decision_tree(best_model, [f'PC{i+1}' for i in range(X_train_pca.shape[1])], 'decision_tree.png')

    # Save the best model and scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')

    print("Metrics DataFrame:\n")
    print(metrics_df)

# Call the main function
if __name__ == '__main__':
    main()
