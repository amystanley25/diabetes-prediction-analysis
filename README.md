### Diabetes Prediction Analysis

**Author**
Amy Stanley

#### Executive Summary
**Goal:**
To develop a machine learning model that accurately predicts whether a patient has diabetes based on clinical features from the Pima Indians Diabetes dataset.

**Challenges:**
The dataset contains missing or invalid entries (e.g., zero values for features like glucose and BMI).
Class imbalance could bias models toward predicting the majority class (non-diabetic).
Ensuring medical relevance and interpretability of the model is critical in healthcare applications.

**Potential Benefits:**
Early detection of diabetes can lead to better clinical outcomes and reduced healthcare costs.
A robust predictive model can assist clinicians in identifying at-risk patients more efficiently.
Highlighting key predictors (like glucose and BMI) can inform screening strategies and patient education.

Through data cleaning, exploratory data analysis (EDA), feature engineering, and model evaluation, we identify key factors affecting diabetes risk and develop multiple classification models. The best-performing model achieves robust predictive performance, highlighting the importance of glucose levels, BMI, and age.

#### Rationale
Diabetes is a prevalent chronic condition with serious health consequences if not detected early. Accurate prediction models can assist healthcare providers in early diagnosis and intervention, potentially improving patient outcomes and reducing healthcare costs.

#### Research Question
Can we accurately predict whether a patient has diabetes based on clinical measurements such as glucose, blood pressure, BMI, and age?


**Model Outcomes or Predictions**
Type of Learning: Classification
Expected Output: Binary prediction:
- 1: Patient has diabetes
- 0: Patient does not have diabetes
Learning Paradigm: Supervised Learning
- The model is trained on labeled data (features and known diabetes outcomes).

#### Data Sources
The dataset used is the publicly available [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database), containing health metrics and diabetes outcomes for female patients of Pima Indian heritage.

#### Methodology

- **Data Cleaning:** Replaced invalid zero values in key clinical features with NaN, then imputed missing values using median values. Removed duplicate records.
- **Exploratory Data Analysis (EDA):** Visualized feature distributions, outliers, and correlations; examined class balance.
- **Feature Engineering:** Created categorical age groups and BMI classes to enhance interpretability.
- **Modeling:** Applied Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine classifiers.
- **Evaluation:** Used accuracy, precision, recall, F1 score, and ROC AUC metrics. Performed hyperparameter tuning on the Random Forest model using GridSearchCV.
- **Visualization:** Plotted ROC curves, precision-recall curves, feature importance, and confusion matrices for model comparison.


#### Results
| Model                  | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression     | 0.753    | 0.667     | 0.618  | 0.642    | 0.821   |
| Decision Tree          | 0.708    | 0.583     | 0.636  | 0.609    | 0.692   |
| Random Forest          | 0.753    | 0.639     | 0.709  | 0.672    | 0.845   |
| Support Vector Machine  | 0.708    | 0.583     | 0.636  | 0.609    | 0.692   |

**Evaluation Metrics:**
Accuracy – Overall correctness of predictions
Precision – How many predicted positives are actual positives
Recall – How many actual positives were correctly predicted (crucial for medical settings)
F1 Score – Harmonic mean of precision and recall
ROC AUC – Measures separability of the model


The Random Forest classifier performed best overall, particularly excelling in recall and ROC AUC, which are crucial for medical diagnosis tasks.

Key predictors identified include **Glucose**, **BMI**, and **Age**.

#### Next steps
- Improve Data Quality: Explore advanced imputation techniques (e.g., K-Nearest Neighbors, multiple imputation) to better handle missing or zero values.
- Feature Engineering: Create new features such as interaction terms, polynomial features, or temporal trends if longitudinal data is available.
- Model Enhancement: Experiment with ensemble methods beyond Random Forest, such as Gradient Boosting Machines (XGBoost, LightGBM, CatBoost), and deep learning models for improved accuracy.
- Hyperparameter Optimization: Use sophisticated tuning methods like RandomizedSearchCV or Bayesian Optimization to efficiently find optimal model parameters.

#### Outline of project
Link to Notebook: https://github.com/amystanley25/diabetes-prediction-analysis/blob/master/proj-notebook.ipynb
- [Set-Up](https://github.com/amystanley25/diabetes-prediction-analysis/blob/master/proj-notebook.ipynb)
- [Data Cleaning & EDA](https://github.com/amystanley25/diabetes-prediction-analysis/blob/master/proj-notebook.ipynb)
- [Modeling](https://github.com/amystanley25/diabetes-prediction-analysis/blob/master/proj-notebook.ipynb)


##### Contact and Further Information
For questions or collaboration opportunities, please contact:  
amystanley2025@gmail.com
