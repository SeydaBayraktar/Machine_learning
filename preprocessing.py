import pandas as pd

data = pd.read_excel("C:/Users/ŞEYDA/Desktop/dataset_chain_store.xlsx")
# ===============================
# 0) DROP EDİLECEK SÜTUNLAR
# ===============================
drop_cols = [
    "Customer demographics",
    "Shipping carriers",
    "Transportation modes",
    "Inspection results",
    "Routes",
    "Supplier name",
    
]

data = data.drop(columns=drop_cols, errors="ignore")

# ===============================
# 1) KATEGORİK SÜTUNLAR
# ===============================
kategorik_sutunlar = [
    "Product type",
    "Machine settings",
    "Machine breakdown",
    "Weather",
    "SKU"
]

data[kategorik_sutunlar] = data[kategorik_sutunlar].astype("category")

# ===============================
# 2) SAYISAL TEMİZLİK
# ===============================
numeric_sutunlar = [
    "Price","Lead times","Shipping times","Shipping costs",
    "Lead time","Manufacturing lead time","Manufacturing costs",
    "Defect rates","Distance",
    "Production line temperature","Order weight",
    "Vehicle capacity","Fuel consumption"
]

for col in numeric_sutunlar:
    data[col] = (
        data[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

# Availability (% → oran)
data["Availability"] = (
    data["Availability"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float) / 100
)

# ===============================
# 3) MANTIKSAL TEMİZLİK
# ===============================

# ❗ Negatif sıcaklıklar SIL
data = data[data["Production line temperature"] >= 0]

# ❗ Defect rate 0–1 aralığı
data["Defect rates"] = data["Defect rates"].clip(0, 1)

# ===============================
# 4) IQR CLIP (SADECE GERÇEKTEN GEREKLİ OLANLAR)
# ===============================
outlier_cols = [
    "Defect rates",
    "Availability",
    "Price","Number of products sold","Revenue generated",
    "Stock levels","Shipping costs","Fuel consumption",
    "Manufacturing costs","Manufacturing lead time","Production line temperature",
    "Order weight","Vehicle capacity"

]

for col in outlier_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    data[col] = data[col].clip(lower, upper)

# ===============================
# 5) EKSİK DEĞER KONTROLÜ
# ===============================
data = data.fillna(data.median(numeric_only=True))

# ===============================
# 6) TEMİZ VERİYİ KAYDET
# ===============================
data.to_excel(
    r"C:\Users\ŞEYDA\Desktop\temizlenmis_veri.xlsx",
    index=False
)

print("Preprocessing tamamlandı. Dosya masaüstüne kaydedildi.")
