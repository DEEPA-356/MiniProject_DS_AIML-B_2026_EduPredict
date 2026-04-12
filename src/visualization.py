import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px

def generate_visualizations(data_path, output_dir):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Attendance vs Final Grade
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='final_grade', y='attendance_percentage', data=df, order=['A', 'B', 'C', 'D', 'F'], palette='muted')
    plt.title('Attendance Percentage by Final Grade')
    plt.xlabel('Final Grade')
    plt.ylabel('Attendance %')
    plt.savefig(os.path.join(output_dir, 'attendance_vs_grade.png'))
    plt.close()
    
    # 2. Performance Label Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='performance_label', data=df, order=['Excellent', 'Good', 'Average', 'At Risk'], palette='viridis')
    plt.title('Distribution of Performance Labels')
    plt.savefig(os.path.join(output_dir, 'performance_distribution.png'))
    plt.close()
    
    # 3. Internal Scores Distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df['internal_score_1'], kde=True, color='blue')
    plt.title('Internal Score 1')

    plt.subplot(1, 3, 2)
    sns.histplot(df['internal_score_2'], kde=True, color='green')
    plt.title('Internal Score 2')

    plt.subplot(1, 3, 3)
    sns.histplot(df['internal_score_3'], kde=True, color='orange')
    plt.title('Internal Score 3')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'internal_scores_hist.png'))
    plt.close()
    
    # 4. Participation vs Performance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='participation_score', y='internal_score_2', hue='performance_label', data=df, alpha=0.7)
    plt.title('Participation Score vs Internal Score 2')
    plt.savefig(os.path.join(output_dir, 'participation_vs_perf.png'))
    plt.close()
    
    # 5. Gender-wise Performance Comparison
    plt.figure(figsize=(10, 6))
    sns.countplot(x='final_grade', hue='gender', data=df, order=['A', 'B', 'C', 'D', 'F'])
    plt.title('Gender-wise Grade Distribution')
    plt.savefig(os.path.join(output_dir, 'gender_grade_dist.png'))
    plt.close()
    
    # 6. Interactive visualization (Plotly)
    try:
        fig = px.sunburst(df, path=['performance_label', 'final_grade', 'gender'], values='age', 
                          title='Performance Breakdown Hierarchy')
        fig.write_image(os.path.join(output_dir, 'sunburst_interactive.png'))
    except Exception as e:
        print(f"Plotly export failed: {e}")
    
    print(f"Visualizations saved successfully to {output_dir}")

if __name__ == "__main__":
    # If running from src, adjust path
    if os.path.basename(os.getcwd()) == 'src':
        generate_visualizations('../dataset/raw_data/student_data.csv', '../outputs/graphs')
    else:
        generate_visualizations('dataset/raw_data/student_data.csv', 'outputs/graphs')
