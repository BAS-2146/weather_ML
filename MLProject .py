#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv ("weather_classification_data (1).csv")


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


mt = df.select_dtypes(include='number').corr()
corr_matrix = mt


# In[6]:


plt.figure(figsize=(11, 11))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# 
# Exploratory data analysis revealed several key relationships among the weather variables. Humidity showed a strong positive correlation with precipitation (r = 0.61), indicating that higher humidity levels are often associated with increased rainfall. Both humidity and precipitation were negatively correlated with visibility (r = -0.48 and -0.46, respectively), which suggests that more moisture in the air tends to reduce visibility. Temperature had a moderate positive correlation with the UV index (r = 0.37), reflecting that warmer days may be linked with higher sun exposure. Wind speed also showed a moderate positive relationship with precipitation (r = 0.44), potentially highlighting storm-related dynamics. Other variables, like atmospheric pressure, had weak correlations overall. These findings help identify which features are likely to be most relevant for further modeling and may also inform decisions around feature selection and preprocessing.
# 
# 

# In[7]:


df.info()


# In[8]:


# Simple outlier check for each numeric column
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers found")


# In[9]:


# Select numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Generate boxplots for all numeric columns
plt.figure(figsize=(10, len(numeric_columns) * 5))
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(len(numeric_columns), 1, i)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


# '''Best approach: Cap outliers (Winsorization)
# Why?
# 
# Weather can have real but rare extremes (e.g., heatwaves, storms).
# 
# Completely removing them might delete valid but rare conditions.
# 
# Capping prevents extreme values from skewing models without losing rows.'''

# In[10]:


# Define a function to handle outliers using the IQR method
def handle_outliers_iqr(data, column, factor=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

# Apply the function to all numeric columns except the last column
numeric_columns = df.select_dtypes(include=['number']).columns[:-1]  # Exclude the last column
for column in numeric_columns:
    df = handle_outliers_iqr(df, column=column)

# Reset the index after removing outliers
df = df.reset_index(drop=True)

# Verify the changes for all numeric columns
plt.figure(figsize=(15, len(df.select_dtypes(include=['number']).columns) * 5))
for i, column in enumerate(df.select_dtypes(include=['number']).columns, 1):
    plt.subplot(len(df.select_dtypes(include=['number']).columns), 1, i)
    sns.boxplot(x=df[column])
    plt.title(f'Box Plot of {column} after outlier handling')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


# In[11]:


# Plot boxplots for numerical columns
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Box Plots for Numerical Features')
plt.show()


# In[12]:


df.describe()


# In[13]:


#Checking for Missing Values
df.isnull().any()


# In[14]:


#cheack if there is a redundant values
df.duplicated().sum()


# In[15]:


df_clean=df.copy()
numeric_df = df_clean.select_dtypes(include='number')

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR)))
print(outliers.sum())


# In[16]:


label_encoder = LabelEncoder()
df['Weather Type'] = label_encoder.fit_transform(df['Weather Type'])


# In[17]:


numerical_cols = df.select_dtypes(include ='number').columns
categorical_cols = df.select_dtypes(exclude ='number').columns


# In[18]:


df_dummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_dummies


# In[19]:


# Get original weather type labels before label encoding
original_weather_types = label_encoder.classes_

# Create pie chart
plt.pie(df['Weather Type'].value_counts().values, autopct='%0.02f%%')

# Use original labels for the legend
plt.legend(original_weather_types)

# Set title using original labels
plt.title('Distribution of Weather Types')

plt.show()


# In[20]:


# Apply dummy encoding to categorical columns
categorical_cols = df.select_dtypes(exclude='number').columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Now split features and labels
X = df_encoded.drop("Weather Type", axis=1)
y = df_encoded["Weather Type"]


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[29]:


# 1. Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_preds_test = rf.predict(X_test)
rf_preds_train = rf.predict(X_train)
# 2. XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_preds_test = xgb.predict(X_test)
xgb_preds_train = xgb.predict(X_train)
# 3. Support Vector Machine (RBF kernel)
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_preds_test = svm.predict(X_test)
svm_preds_train = svm.predict(X_train)
# 4. Multilayer Perceptron (MLPClassifier)
mlp = MLPClassifier(max_iter=300)
mlp.fit(X_train, y_train)
mlp_preds_test = mlp.predict(X_test)
mlp_preds_train = mlp.predict(X_train)


# In[30]:


# Random Forest Evaluation
print("--- Random Forest ---")
print(f"Test Accuracy: {accuracy_score(y_test,rf_preds_test):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train,rf_preds_train):.4f}")
print(f"Precision: {precision_score(y_test, rf_preds_test, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, rf_preds_test, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, rf_preds_test, average='weighted'):.4f}")
print("\n")

# XGBoost Evaluation
print("--- XGBoost ---")
print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds_test):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train, xgb_preds_train):.4f}")
print(f"Precision: {precision_score(y_test, xgb_preds_test, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, xgb_preds_test, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, xgb_preds_test, average='weighted'):.4f}")
print("\n")

# SVM (RBF) Evaluation
print("--- SVM (RBF Kernel) ---")
print(f"Test Accuracy: {accuracy_score(y_test, svm_preds_test):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train, svm_preds_train):.4f}")
print(f"Precision: {precision_score(y_test, svm_preds_test, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, svm_preds_test, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, svm_preds_test, average='weighted'):.4f}")
print("\n")

# MLP Evaluation
print("--- MLPClassifier ---")
print(f"Test Accuracy: {accuracy_score(y_test, mlp_preds_test):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train, mlp_preds_train):.4f}")
print(f"Precision: {precision_score(y_test, mlp_preds_test, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, mlp_preds_test, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, mlp_preds_test, average='weighted'):.4f}")
print("\n")


# In[31]:


# Scale numerical features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# In[32]:


# Split into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)


# In[35]:


# 1. Random Forest Classifier
rf2 = RandomForestClassifier()
rf2.fit(X_train2, y_train2)
rf_preds_test2 = rf2.predict(X_test2)
rf_preds_train2 = rf2.predict(X_train2)
# 2. XGBoost Classifier
xgb2 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb2.fit(X_train2, y_train2)
xgb_preds_test2 = xgb2.predict(X_test2)
xgb_preds_train2 = xgb2.predict(X_train2)
# 3. Support Vector Machine (RBF kernel)
svm2 = SVC(kernel='rbf')
svm2.fit(X_train2, y_train2)
svm_preds_test2 = svm2.predict(X_test2)
svm_preds_train2 = svm2.predict(X_train2)
# 4. Multilayer Perceptron (MLPClassifier)
mlp2 = MLPClassifier(max_iter=300)
mlp2.fit(X_train2, y_train2)
mlp_preds_test2 = mlp2.predict(X_test2)
mlp_preds_train2 = mlp2.predict(X_train2)


# In[36]:


# Random Forest Evaluation
print("--- Random Forest ---")
print(f"Test Accuracy: {accuracy_score(y_test2,rf_preds_test2):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train2,rf_preds_train2):.4f}")
print(f"Precision: {precision_score(y_test2, rf_preds_test2, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test2, rf_preds_test2, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test2, rf_preds_test2, average='weighted'):.4f}")
print("\n")

# XGBoost Evaluation
print("--- XGBoost ---")
print(f"Test Accuracy: {accuracy_score(y_test2, xgb_preds_test2):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train2, xgb_preds_train2):.4f}")
print(f"Precision: {precision_score(y_test2, xgb_preds_test2, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test2, xgb_preds_test2, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test2, xgb_preds_test2, average='weighted'):.4f}")
print("\n")

# SVM (RBF) Evaluation
print("--- SVM (RBF Kernel) ---")
print(f"Test Accuracy: {accuracy_score(y_test2, svm_preds_test2):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train2, svm_preds_train2):.4f}")
print(f"Precision: {precision_score(y_test2, svm_preds_test2, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test2, svm_preds_test2, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test2, svm_preds_test2, average='weighted'):.4f}")
print("\n")

# MLP Evaluation
print("--- MLPClassifier ---")
print(f"Test Accuracy: {accuracy_score(y_test2, mlp_preds_test2):.4f}")
print(f"Train Accuracy: {accuracy_score(y_train2, mlp_preds_train2):.4f}")
print(f"Precision: {precision_score(y_test2, mlp_preds_test2, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test2, mlp_preds_test2, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test2, mlp_preds_test2, average='weighted'):.4f}")
print("\n")


# In[37]:


#Define parameter grids
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,), (100, 100)],
        'activation': ['relu', 'tanh'],
        'max_iter': [300]
    }
}

#Create GridSearchCV objects for each model
rf_grid_search = GridSearchCV(RandomForestClassifier(), param_grids['RandomForest'], cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_grids['XGBoost'], cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search = GridSearchCV(SVC(), param_grids['SVM'], cv=5, scoring='accuracy', n_jobs=-1)
mlp_grid_search = GridSearchCV(MLPClassifier(), param_grids['MLP'], cv=5, scoring='accuracy', n_jobs=-1)


# In[38]:


# Fit the models
rf_grid_search.fit(X_train2, y_train2)
xgb_grid_search.fit(X_train2, y_train2)
svm_grid_search.fit(X_train2, y_train2)
mlp_grid_search.fit(X_train2, y_train2)

# Print best parameters
print("Best RF:", rf_grid_search.best_params_)
print("Best XGB:", xgb_grid_search.best_params_)
print("Best SVM:", svm_grid_search.best_params_)
print("Best MLP:", mlp_grid_search.best_params_)


# In[39]:


# Reuse the best parameters found from GridSearchCV
best_rf = RandomForestClassifier(**rf_grid_search.best_params_).fit(X_train2, y_train2)
best_xgb = XGBClassifier(**xgb_grid_search.best_params_, use_label_encoder=False, eval_metric='mlogloss').fit(X_train2, y_train2)
best_svm = SVC(**svm_grid_search.best_params_).fit(X_train2, y_train2)
best_mlp = MLPClassifier(**mlp_grid_search.best_params_).fit(X_train2, y_train2)

# Predictions and Accuracy
print("Best RF Teast Accuracy:", accuracy_score(y_test2, best_rf.predict(X_test2)))
print("Best RF Trainn Accuracy:", accuracy_score(y_train2, best_rf.predict(X_train2)))

print("Best XGB Test Accuracy:", accuracy_score(y_test2, best_xgb.predict(X_test2)))
print("Best XGB Train Accuracy:", accuracy_score(y_train2, best_xgb.predict(X_train2)))

print("Best SVM Test Accuracy:", accuracy_score(y_test2, best_svm.predict(X_test2)))
print("Best SVM Train Accuracy:", accuracy_score(y_train2, best_svm.predict(X_train2)))

print("Best MLP Test Accuracy:", accuracy_score(y_test2, best_mlp.predict(X_test2)))
print("Best MLP Train Accuracy:", accuracy_score(y_train2, best_mlp.predict(X_train2)))


# In[40]:


models = {
    "RandomForest": best_rf,
    "SVM": best_svm,
    "XGBoost": best_xgb,
    "MLP": best_mlp
}

for name, model in models.items():
    preds3 = model.predict(X_test2)
    print(f"\n{name} Metrics")
    print("Precision:", precision_score(y_test2, preds3, average='weighted'))
    print("Recall   :", recall_score(y_test2, preds3, average='weighted'))
    print("F1 Score :", f1_score(y_test2, preds3, average='weighted'))


# In[41]:


import joblib
joblib.dump(best_rf, 'best_rf.pkl')


# In[42]:


import os
print("Saved at:", os.path.abspath('best_rf.pkl'))

