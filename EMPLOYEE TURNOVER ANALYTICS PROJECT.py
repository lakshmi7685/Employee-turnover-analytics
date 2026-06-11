# ==========================================
# EMPLOYEE TURNOVER ANALYTICS PROJECT
# ==========================================

# =========================
# Import Libraries
# =========================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE

# =========================
# Load Dataset
# =========================

df = pd.read_csv("HR_comma_sep.csv")

print("First 5 Rows")
print(df.head())

# =========================
# 1. Data Quality Checks
# =========================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates if needed
df = df.drop_duplicates()

# =========================
# 2. EDA Analysis
# =========================

# -------------------------
# 2.1 Correlation Heatmap
# -------------------------

plt.figure(figsize=(12,8))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# 2.2 Distribution Plots
# -------------------------

plt.figure(figsize=(6,4))
sns.histplot(df['satisfaction_level'], kde=True)
plt.title("Employee Satisfaction Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['last_evaluation'], kde=True)
plt.title("Employee Evaluation Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['average_montly_hours'], kde=True)
plt.title("Average Monthly Hours Distribution")
plt.show()

# -------------------------
# 2.3 Project Count vs Left
# -------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x='number_project',
    hue='left',
    data=df
)

plt.title("Project Count vs Employee Turnover")
plt.show()

# =========================
# 3. KMeans Clustering
# =========================

# Employees who left
left_emp = df[df['left'] == 1]

cluster_data = left_emp[
    ['satisfaction_level', 'last_evaluation']
]

# KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

left_emp['cluster'] = kmeans.fit_predict(cluster_data)

# Plot clusters
plt.figure(figsize=(8,6))

sns.scatterplot(
    x='satisfaction_level',
    y='last_evaluation',
    hue='cluster',
    data=left_emp,
    palette='Set1'
)

plt.title("Employee Clusters")
plt.show()

# =========================
# 4. Data Preprocessing
# =========================

# Separate features and target
X = df.drop('left', axis=1)
y = df['left']

# Categorical columns
cat_cols = ['Department', 'salary']

# Numerical columns
num_cols = X.drop(cat_cols, axis=1)

# Convert categorical to dummy
cat_data = pd.get_dummies(
    X[cat_cols],
    drop_first=True
)

# Combine
X_final = pd.concat(
    [num_cols, cat_data],
    axis=1
)

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    stratify=y,
    random_state=123
)

# =========================
# 4.3 SMOTE
# =========================

smote = SMOTE(random_state=123)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# =========================
# 5. Model Training
# =========================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(random_state=123),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=123)
}

results = {}

# =========================
# Cross Validation
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

for name, model in models.items():

    print("\n================================")
    print(name)
    print("================================")

    # Cross validated predictions
    y_pred_cv = cross_val_predict(
        model,
        X_train_smote,
        y_train_smote,
        cv=cv
    )

    print("\nClassification Report:")
    print(classification_report(
        y_train_smote,
        y_pred_cv
    ))

    # Train model
    model.fit(X_train_smote, y_train_smote)

    # Test prediction
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:,1]

    # ROC AUC
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model,
        'auc': auc,
        'pred': y_pred,
        'prob': y_prob
    }

    print("\nROC AUC:", auc)

# =========================
# 6. ROC Curve
# =========================

plt.figure(figsize=(8,6))

for name in results:

    fpr, tpr, _ = roc_curve(
        y_test,
        results[name]['prob']
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={results[name]['auc']:.3f}"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# Confusion Matrices
# =========================

for name in results:

    print("\n========================")
    print(name)
    print("========================")

    cm = confusion_matrix(
        y_test,
        results[name]['pred']
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()

# =========================
# Best Model
# =========================

best_model_name = max(
    results,
    key=lambda x: results[x]['auc']
)

best_model = results[best_model_name]['model']

print("\nBest Model:", best_model_name)

# =========================
# 7. Employee Risk Zones
# =========================

turnover_prob = best_model.predict_proba(X_test)[:,1]

risk_df = X_test.copy()

risk_df['Turnover_Probability'] = turnover_prob

# Risk Categories
def risk_zone(prob):

    if prob < 0.20:
        return "Safe Zone"

    elif prob < 0.60:
        return "Low Risk Zone"

    elif prob < 0.90:
        return "Medium Risk Zone"

    else:
        return "High Risk Zone"

risk_df['Risk_Zone'] = risk_df[
    'Turnover_Probability'
].apply(risk_zone)

print("\nEmployee Risk Zones:")
print(
    risk_df[
        ['Turnover_Probability', 'Risk_Zone']
    ].head()
)

# =========================
# Retention Strategies
# =========================

print("\nRetention Strategies")

print("""
1. Safe Zone:
   - Reward and recognize employees
   - Continue engagement programs

2. Low Risk Zone:
   - Conduct regular feedback sessions
   - Offer career growth opportunities

3. Medium Risk Zone:
   - Reduce workload stress
   - Improve manager communication
   - Review compensation

4. High Risk Zone:
   - Immediate HR intervention
   - Salary revision
   - Promotion opportunities
   - Personalized retention plans
""")