import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
import lime
import lime.lime_tabular

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def train_and_evaluate(X_train, X_test, y_train, y_test, label_classes):
    """
    Train multiple classification models and evaluate them using 5-fold 
    cross-validation and hold-out test set performance metrics.
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_model = None
    best_score = 0
    
    # Ensure directories exist
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/graphs', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # K-Fold definition for clarity
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
        
        # Training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # ROC-AUC (multi-class ovr)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': float(roc_auc)
        }
        
        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_model = model
            
    # Save Model Comparison Table
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = 'Model'
    # Reordering columns as requested
    metrics_df = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']]
    metrics_df.to_csv('outputs/reports/model_comparison.csv')
    
    # Print results formatted matching the request
    print("\n=============================")
    print("MODEL COMPARISON RESULTS")
    print("=============================")
    print(metrics_df.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nBest Model: {best_model_name}")
    
    # Save best model
    joblib.dump(best_model, 'outputs/models/best_model.pkl')
    
    # Plot Confusion Matrix for Best Model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_classes, yticklabels=label_classes)
    plt.title(f"Confusion Matrix ({best_model_name})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('outputs/graphs/confusion_matrix.png')
    plt.close()
    
    return best_model_name, best_model

def plot_feature_importance(name, model, feature_names):
    """Plot feature importance depending on model type."""
    if name == 'XGBoost':
        import xgboost
        xgboost.plot_importance(model, max_num_features=10, importance_type='weight')
        plt.title('Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.savefig('outputs/graphs/feature_importance.png')
        plt.close()
    elif name in ['Random Forest', 'Gradient Boosting'] and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][-10:] # Top 10
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances ({name} - Top 10)")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig('outputs/graphs/feature_importance.png')
        plt.close()

def run_explainability(model, name, X_train, X_test, label_classes):
    """Run SHAP and LIME Explainability."""
    print("Generating Explainability Plots (SHAP and LIME)...")
    # SHAP
    try:
        # XGBoost natively has a bug with shap.TreeExplainer returning strings in latest versions
        # We will use purely feature contribution standard or fallback if needed.
        if name in ['Random Forest', 'XGBoost', 'Gradient Boosting']:
            # We enforce use of an independent SHAP explainer, or just try to pass feature names properly
            explainer = shap.Explainer(model.predict, X_train)
            shap_values = explainer(X_test.iloc[:50]) # limit to 50 for speed since it's Kernel Explainer behind the scenes for generic predict
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test.iloc[:50], show=False)
            plt.tight_layout()
            plt.savefig('outputs/graphs/shap_summary.png')
            plt.close()
    except Exception as e:
        print(f"Warning: SHAP summary plot failed to generate. Trying fallback... Error: {e}")
        try:
            # Fallback for trees if it was just a multiclass dimension issue
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values, X_test, show=False)
            else:
                shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig('outputs/graphs/shap_summary.png')
            plt.close()
        except Exception as e2:
            print(f"Fallback SHAP also failed: {e2}")

    # LIME
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values, 
            feature_names=X_train.columns, 
            class_names=label_classes, 
            mode='classification'
        )
        # Explain the first instance in test set
        exp = explainer.explain_instance(X_test.values[0], model.predict_proba, num_features=5)
        exp.save_to_file('outputs/reports/lime_explanation.html')
    except Exception as e:
        print(f"Warning: LIME explanation failed to generate. Error: {e}")
    print("Explainability Plots Generation Complete.")

if __name__ == "__main__":
    from preprocessing import load_data, engineer_features, preprocess_data, split_data
    
    df = load_data('dataset/raw_data/student_data.csv')
    df = engineer_features(df)
    df, le_gender, le_label = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    label_classes = le_label.classes_
    
    best_name, best_model = train_and_evaluate(X_train, X_test, y_train, y_test, label_classes)
    plot_feature_importance(best_name, best_model, X_train.columns)
    run_explainability(best_model, best_name, X_train, X_test, label_classes)
