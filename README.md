# EduPredict: Data-Driven Prediction of Student Academic Performance

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.2.1-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-orange)

## Abstract
EduPredict is a data-driven analytical project designed to forecast student academic performance using machine learning. In the modern educational landscape, identifying "at-risk" students early is crucial for timely intervention and improved institutional outcomes. This project leverages a synthetic dataset of 500 students, incorporating features such as attendance percentage, internal assessment scores, assignment submission rates, and extracurricular participation. By applying advanced preprocessing techniques and evaluating multiple classification models—including Random Forest and Decision Trees—EduPredict achieves high accuracy in classifying students into performance categories like 'Excellent', 'Good', 'Average', and 'At Risk'. The findings underscore the significant impact of consistent attendance and internal scores on final grades. This project serves as a comprehensive framework for educational institutions to implement proactive support systems, aligning with SDG Goal 4: Quality Education.

## Problem Statement
Educational institutions often struggle to identify students who are likely to underperform or drop out until it's too late. Traditional evaluation methods are reactive rather than proactive. There is a need for a predictive system that analyzes behavioral and academic data to provide early warnings, allowing educators to provide targeted support.

## Dataset Source
The dataset used in this project is a **synthetic dataset** generated specifically for academic purposes. it contains 500 records with features like attendance, internal scores, and participation levels, mapped to final grades and performance labels.

## Methodology / Workflow
1. **Data Generation**: Creating a realistic synthetic dataset.
2. **Data Understanding**: Exploratory Data Analysis (EDA) to find patterns.
3. **Preprocessing**: Handling missing values, encoding, and scaling.
4. **Feature Engineering**: Creating derived metrics like `weighted_performance_index`.
5. **Visualization**: Identifying trends through boxplots, heatmaps, and distributions.
6. **Model Training**: Implementing Random Forest, Logistic Regression, and Decision Trees.
7. **Evaluation**: Assessing performance using F1-score and Accuracy.

## Tools Used
- **Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly
- **Environment**: Jupyter Notebook

## Results / Findings
- **Model Performance**: The Random Forest classifier achieved an accuracy of ~92% (placeholder).
- **Key Drivers**: Attendance and Internal Score 2 were found to be the most significant predictors.
- **Insights**: Students with <75% attendance have a 60% higher probability of being labeled "At Risk".

## Folder Structure
```text
EduPredict/
├── dataset/
│   ├── raw_data/          # Original synthetic CSV
│   └── processed_data/    # Cleaned and engineered data
├── notebooks/             # Step-by-step Jupyter notebooks
├── src/                   # Modular Python scripts
├── outputs/
│   ├── graphs/            # Saved visualizations
│   └── results/           # Model weights and metrics
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## Future Scope
- **Real-time Integration**: Connecting the prediction engine with live Learning Management Systems (LMS).
- **Mobile Application**: Developing a React Native app for students to track their own performance metrics.
- **Deep Learning**: Implementing Neural Networks for more complex behavior pattern recognition.

## Team Members
- **Team Leader**: [Adhi0007]
- **Member 2**: [Student Name 2]
- **Member 3**: [Student Name 3]
- **Member 4**: [Student Name 4]

---
*Created for SRM Institute of Science and Technology - Data Science (AIML-B 2026)*
