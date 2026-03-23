import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(filepath):
    """Load the student dataset."""
    return pd.read_csv(filepath)

def engineer_features(df):
    """Create derived features like average scores and weighted index."""
    df = df.copy()
    df['avg_internal_score'] = (df['internal_score_1'] + df['internal_score_2'] + df['internal_score_3']) / 3
    
    # Weighted index based on literature
    df['weighted_performance_index'] = (
        df['avg_internal_score'] * 0.4 + 
        df['attendance_percentage'] * 0.3 + 
        df['assignment_submission_rate'] * 0.2 + 
        df['participation_score'] * 1.0
    )
    return df

def preprocess_data(df):
    """Encode categorical variables and scale numerical features."""
    df = df.copy()
    
    # Label Encoding
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    
    le_label = LabelEncoder()
    df['performance_label'] = le_label.fit_transform(df['performance_label'])
    
    # Scaling
    features_to_scale = [
        'age', 'attendance_percentage', 'internal_score_1', 'internal_score_2', 
        'internal_score_3', 'assignment_submission_rate', 'participation_score', 
        'library_usage_hours', 'extracurricular_score', 'avg_internal_score', 'weighted_performance_index'
    ]
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df, le_gender, le_label

def split_data(df, target='performance_label', test_size=0.2):
    """Split data into training and testing sets."""
    X = df.drop(['student_id', 'name', 'final_grade', target], axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

if __name__ == "__main__":
    df = load_data('dataset/raw_data/student_data.csv')
    df = engineer_features(df)
    df, _, _ = preprocess_data(df)
    
    os.makedirs('dataset/processed_data', exist_ok=True)
    df.to_csv('dataset/processed_data/processed_student_data.csv', index=False)
    print("Preprocessing complete. Processed data saved.")
