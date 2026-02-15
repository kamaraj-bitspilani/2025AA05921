# ML Assignment 2: Loan Approval Classification

## **a. Problem Statement**

The  objective of this assignment is to  develop a machine learning model that can predict whether a loan application will be approved or rejected. Using the previous loan data, the model will learn patterns and relationships between various attributes and past approval decisions. The dataset includes information such as demographic details, financial status, credit history, and specific loan attributes. Based on these features, the binary classification model that can accurately evaluate and classify new, unseen loan applications.

## **b. Dataset Description**

The dataset used for this assignment is a loan approval classification dataset containing historical loan application records with various applicants.

**Dataset Source:** Kaggle

**Dataset Link:** [Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data?resource=download)


### Dataset Overview:
- **Total Instances:** 45,000 loan applications 
- **Total Features:** 13 features + 1 Target variable
- **Classification Type:** Binary classification (Approved/Rejected)
- **Data Source:** KAGGLE
- **Format:** CSV (loan_data.csv)

### Feature Description:

1. **person_age** (Numeric) - Age of the loan applicant in years
2. **person_gender** (Categorical) - Gender of applicant (Male/Female)
3. **person_education** (Categorical) - Highest education level (High School, Bachelor, Master, Associate, Doctorate)
4. **person_income** (Numeric) - Annual income in dollars
5. **person_emp_exp** (Numeric) - Years of employment experience
6. **person_home_ownership** (Categorical) - Housing status (RENT, OWN, MORTGAGE, OTHER)
7. **loan_amnt** (Numeric) - Requested loan amount in dollars
8. **loan_intent** (Categorical) - Purpose of the loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOME_IMPROVEMENT, DEBT_CONSOLIDATION)
9. **loan_int_rate** (Numeric) - Interest rate as percentage
10. **loan_percent_income** (Numeric) - Loan amount as percentage of annual income
11. **cb_person_cred_hist_length** (Numeric) - Length of credit history in years
12. **credit_score** (Numeric) - Credit score of the applicant (300-850 range)
13. **previous_loan_defaults_on_file** (Categorical) - History of previous defaults (Yes/No)

### Target Variable:
- **loan_status** (Binary) - Loan approval outcome (0 = Rejected, 1 = Approved)

### Data Characteristics:
- **Missing Values:** No Missing values 
- **Class Distribution:** Imbalanced classification (majority class: Approved) (Stratified sampling will be done)
- **Categorical Features:** 5 categorical features (Encoding will be done )
- **Numerical Features:** 8 numerical features with varying scales (Scaling will be done)

## **c. Models Used** 

This assignment implements and evaluates **six different classification models** 
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

### Model Comparison Table for all metrics

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8968 | 0.9515 | 0.8953 | 0.8968 | 0.8959 | 0.6969 |
| Decision Tree | 0.9160 | 0.9600 | 0.9146 | 0.9160 | 0.9125 | 0.7455 |
| K-Nearest Neighbors | 0.8986 | 0.9518 | 0.8139 | 0.7048 | 0.7554 | 0.6947 |
| Naive Bayes | 0.8132 | 0.7920 | 0.6892 | 0.2900 | 0.4082 | 0.3607 |
| Random Forest | 0.9283 | 0.9735 | 0.8946 | 0.7676 | 0.8263 | 0.7850 |
| XGBoost | 0.9348 | 0.9774 | 0.9039 | 0.7904 | 0.8434 | 0.8053 |

### **Best Performing Model**
**XGBoost** with **93.48% accuracy** and **0.8053 MCC**

---

### **Model Performance Observations**

#### 1. Logistic Regression
Logistic regression achieves strong performance (89.68% accuracy, 0.6969 MCC) as a linear baseline model, indicating that loan approval data follows the linear pattern. High AUC (0.9515) score indicates strong class discrimination between approved and rejected loans. And Precision and recall values are almost equal, so the model does not seem biased toward one class.And the  advantage is interpretability of the model that we can understand how each feature influences the decision of the loan approval. 

#### 2. Decision Tree
The Decision Tree improved the performance compared to previous Logistic Regression model  and it achieved  the performance (91.60% accuracy, 0.7455 MCC)  through decision rules.  Precision and recall are both balanced, showing stable predictions across both classes. One useful aspect of Decision Trees is the ability to visualize decision rules and identify important features. 

#### 3. K-Nearest Neighbors
KNN model achieved performance similar to Logistic Regression only in terms of accuracy. The AUC score is also strong, meaning the model distinguishes classes reasonably well.Achieves good performance (89.86% accuracy, 0.6947 MCC) with proper feature scaling. Distance-based approach captures local patterns in loan approval decision-making with excellent AUC (0.9518). However, shows imbalanced performance: high precision (0.8139) but lower recall (0.7048), Since the KNN model depends heavily on distance calculations, feature scaling was very important here. While it performs decently, it does not outperform tree-based ensemble methods.

#### 4. Naive Bayes
Naive Bayes model provides the low performance among all other models. Eventhough  the accuracy might look acceptable but the recall value is very low indicates that the  model fails to correctly classify many approved loans. The naive bayes model achieved (81.32% accuracy, 0.3607 MCC) and its worst among all other models as discussed and  extremely low recall (0.2900) and F1-score (0.4082) reveal severe class imbalance handling issues.Lowest AUC (0.7920) and MCC (0.3607) confirm poor  discrimination capability.  Fast training is its only advantage and its Not recommended for loan approval classification problem.

#### 5. Random Forest (Ensemble)
Random Forest model performed very well compared to the single models like Decision Tree and Logistic Regression. The ensemble approach(combining models) reduces overfitting by averaging multiple trees output and do the majority voting and take the final decision based on the Various model results .It delivers outstanding performance  (92.83% accuracy, 0.7850 MCC) through ensemble averaging of multiple trees. And its a Second-best model overall with excellent AUC (0.9735) demonstrating strong discrimination capability. High precision (0.8946) with moderate recall (0.7676) indicates conservative but accurate approval predictions. Through bagging it reduces the overfitting issue. 

#### 6. XGBoost (Ensemble)
XGBoost model  provided the best overall performance among all the 6 models. XGBoost achieved the highest accuracy, AUC, and MCC score. And it represents that boosting helps in learning complex patterns by correcting previous errors step by step gradually . From the value point of view , it achieves (93.48% accuracy, 0.8053 MCC) through sequential gradient boosting optimization. it handles the imbalanced classification data very well. Excellent precision (0.9039) with good recall (0.7904) provides balanced performance. **Strongly recommended as primary model for loan approval classification.**

---

