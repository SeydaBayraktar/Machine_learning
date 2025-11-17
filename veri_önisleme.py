import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
plt.ion()
warnings.filterwarnings('ignore')

#1. Libraries and Dataset Overview

# Load the dataset
data = pd.read_csv("C:/Users/ŞEYDA/Desktop/dataset.csv", sep=';', decimal=',')
print(data.head())
print(data.columns)

# 2. initial  exploration

data.info()

# Check unique values for categorical columns
non_numeric_columns=data.select_dtypes(exclude=["number"]).columns
for col in non_numeric_columns:
    print(f"{col} : {data[col].unique()}")

#3.Data Cleaning

data = data.drop('SKU', axis=1)
#data = data[~data['Customer demographics'].isin(['Non-binary', 'Unknown'])]
data=data[data['Inspection results'] != 'Pending']

#4. Feature Engineering

data = data.drop(['Lead times','Shipping times','Shipping carriers','Shipping costs','Transportation modes','Routes',
                   'Price',	 'Number of products sold','Revenue generated','Stock levels','Lead times',	
                    'Lead time', 'Production volumes', 'Costs','Lead time', 'Production volumes',	               
], axis=1)

#5. Visualization

non_numeric_columns = data.select_dtypes(exclude=["number"]).columns
#Correlation matrix
correlation_matrix = data.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Count plots for all categorical columns
def plot_countplots(data, columns, hue=None, figsize=(12, 30), bins_dict=None):  
    n = len(columns)
    fig, axs = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axs = [axs]
    
    plt.tight_layout(pad=6)
    
    for i, col in enumerate(columns):
            plot_data = data.copy()
            
            if bins_dict and col in bins_dict:
                bins = bins_dict[col]
                plot_data[f'{col} Group'] = pd.cut(plot_data[col], bins=bins)
                x_col = f'{col} Group'
            else:
                x_col = col
            sns.countplot(ax=axs[i], x=x_col, hue=hue, data=plot_data, palette='viridis' if bins_dict and col in bins_dict else None)
            axs[i].set_title(f'Distribution by {col}' + (f' and {hue}' if hue else ''))
            axs[i].set_xlabel(col if not (bins_dict and col in bins_dict) else f'{col} Group')
            axs[i].set_ylabel('Count')
            axs[i].tick_params(axis='x', rotation=45)

    
    plt.show()

plot_countplots(data, non_numeric_columns, 'Inspection results')

#6. Encoding and Scaling

# One-hot encoding
data_encoded = pd.get_dummies(
    data,
    columns=['Product type', 'Customer demographics', 'Supplier name'],
    drop_first=False
)
print(data_encoded.columns)
# Features and target
feature_cols = [ 
    'Defect rates',
    'Manufacturing lead time',
    'Manufacturing costs',
    #'Lead time',
    #'Production volumes',
    'Availability',
    'Order quantities',
    'Product type_skincare',
    'Product type_cosmetics',
    'Customer demographics_Female',
    'Supplier name_Supplier 4',
    'Supplier name_Supplier 5',
    'Supplier name_Supplier 3'
    
]
X = data_encoded[feature_cols]
y = LabelEncoder().fit_transform(data['Inspection results'])
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#7. ModelEvaluator Class: Multiple Model Training

class ModelEvaluator:
    def __init__(self, models: dict):
        self.modelsorm = models

    def train_evaluate(self, X_train, y_train, X_test, y_test):
        for name, model in self.modelsorm.items():
            print(f"\n{'='*30}\nModel: {name}\n{'='*30}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            print(classification_report(y_test, y_pred))
            print("Accuracy:", accuracy_score(y_test, y_pred))

            if y_prob is not None:
                roc_auc = roc_auc_score(y_test, y_prob)
                print(f"ROC AUC: {roc_auc:.4f}")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.figure()
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                plt.plot([0,1], [0,1], 'k--')
                plt.title(f'ROC Curve - {name}')
                plt.legend()
                plt.show()

            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.show()
plt.show()
# Model list
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "GBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}
print(models)
evaluator = ModelEvaluator(models)
evaluator.train_evaluate(X_train, y_train, X_test, y_test)
#plt.show(block=True)
