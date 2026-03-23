import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple models and return metrics."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    os.makedirs('outputs/results', exist_ok=True)
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'CV Accuracy': np.mean(cv_scores),
            'Test Accuracy': acc,
            'F1-Score': report['weighted avg']['f1-score']
        }
        
        print(f"{name} - Test Accuracy: {acc:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_model = (name, model)
            
    # Save metrics
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv('outputs/results/model_metrics.csv')
    
    # Save best model
    joblib.dump(best_model[1], 'outputs/results/best_model.pkl')
    print(f"\nBest model: {best_model[0]} saved to outputs/results/best_model.pkl")
    
    return best_model

def plot_feature_importance(model, feature_names):
    """Plot feature importance for a fitted model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('outputs/graphs/feature_importance.png')
        plt.show()

if __name__ == "__main__":
    from preprocessing import load_data, engineer_features, preprocess_data, split_data
    
    df = load_data('dataset/raw_data/student_data.csv')
    df = engineer_features(df)
    df, _, _ = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    
    best_name, best_model = train_and_evaluate(X_train, X_test, y_train, y_test)
    plot_feature_importance(best_model, X_train.columns)
