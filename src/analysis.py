import pandas as pd
import numpy as np

def get_summary_stats(df):
    """Return basic descriptive statistics."""
    return df.describe()

def detect_outliers_iqr(df, column):
    """Detect outliers using the Interquartile Range (IQR) method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def get_correlation_matrix(df):
    """Calculate correlation matrix for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()

def filter_at_risk_students(df):
    """Return a subset of students who are labeled as 'At Risk'."""
    return df[df['performance_label'] == 'At Risk']

if __name__ == "__main__":
    df = pd.read_csv('dataset/raw_data/student_data.csv')
    print("Summary Statistics:\n", get_summary_stats(df))
    
    attendance_outliers = detect_outliers_iqr(df, 'attendance_percentage')
    print(f"\nDetected {len(attendance_outliers)} attendance outliers.")
    
    at_risk = filter_at_risk_students(df)
    print(f"\nFound {len(at_risk)} at-risk students.")
