import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =======================
# 1. Veri Yükleme
# =======================
data = pd.read_excel("C:/Users/ŞEYDA/Desktop/temizlenmis_veri.xlsx")
data = data.drop(columns=["SKU"], errors="ignore")


# =======================
# 2. Target: Gecikme (%75 persentil)
# =======================
threshold = data["Lead times"].quantile(0.75)
data["Delayed"] = np.where(data["Lead times"] > threshold, 1, 0)


# =======================
# FEATURE ENGINEERING
# =======================

# =======================
# 5. Product type one-hot
# =======================
data_encoded = pd.get_dummies(
    data,
    columns=["Weather"],
    drop_first=False
)
print(data_encoded.columns)
# =======================
# 6. Feature Set +
# =======================
feature_cols = [
    "Revenue generated",
    "Number of products sold",
    "Order quantities",
    "Production volumes",
    "Manufacturing lead time",
    

    "Weather_Snowy",
    "Weather_Rainy",

]

X = data_encoded[feature_cols]
y = data_encoded["Delayed"]

# =======================
# 7. Train / Test
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =======================
# 8. Scaling
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 9. Modeller
# =======================
models = {
      "KNN": KNeighborsClassifier(
        n_neighbors=9,
        weights="distance"
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=False
    )
}

# =======================
# 10. Eğitim & Değerlendirme
# =======================
for name, model in models.items():
    print("\n==========================")
    print(f"MODEL: {name}")
    print("==========================")

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# CatBoost modelini tekrar al (aynı ayarlar)
cat_model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    auto_class_weights="Balanced",
    random_state=42,
    verbose=False
)
# Eğit
cat_model.fit(X_train, y_train)

# Test seti tahminleri
y_pred_cat = cat_model.predict(X_test)
y_prob_cat = cat_model.predict_proba(X_test)[:, 1]

# Sonuç DataFrame'i
cat_results = X_test.copy()
cat_results["Actual_Delayed"] = y_test.values
cat_results["Predicted_Delayed"] = y_pred_cat
cat_results["Delayed_Probability"] = y_prob_cat

# Masaüstüne kaydet
output_path = "C:/Users/ŞEYDA/Desktop/catboost_results.xlsx"
cat_results.to_excel(output_path, index=False)

print("\n📌 CatBoost sonuçları masaüstüne kaydedildi:")
print(output_path)
