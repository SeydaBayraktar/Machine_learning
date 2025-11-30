import pandas as pd
from ctgan import CTGAN
from scipy.stats import chi2_contingency

# 1) Veri setini yükle
data = pd.read_csv("C:/Users/ŞEYDA/Desktop/supply_chain_data.csv", sep=",")

# 2) Kategorik kolonlar
discrete_columns = [
    "Product type",
    "SKU",
    "Customer demographics",
    "Shipping carriers",
    "Supplier name",
    "Transportation modes",
    "Routes",
    "Inspection results",
    "Location"
]

# 3) CTGAN modeli
ctgan = CTGAN(epochs=300)
ctgan.fit(data, discrete_columns)

# 4) 200 yeni sentetik veri üret
new_data = ctgan.sample(300)

# 5) Yeni veri ile eski veriyi birleştir
augmented_data = pd.concat([data, new_data], ignore_index=True)

# 6) Sonuçları göster
print("\n--- Orijinal sınıf dağılımı ---")
print(data["Inspection results"].value_counts())

print("\n--- Yeni üretilen verinin sınıf dağılımı ---")
print(new_data["Inspection results"].value_counts())

print("\n--- Birleştirilmiş veri seti sınıf dağılımı ---")
print(augmented_data["Inspection results"].value_counts())

# 7) Yeni veri setini kaydet
augmented_data.to_csv("C:/Users/ŞEYDA/Desktop/dataset_augmented.csv", index=False)
print("\nYeni veri seti 'dataset_augmented.csv' olarak kaydedildi!")

# --------------------------------------------------------
# 8) Chi-Square (χ²) Testi — Kategorik kolonların dağılım benzerliği
# --------------------------------------------------------

categorical_cols = discrete_columns  # aynı listeyi kullanıyoruz

print("\n\n==============================")
print("   CHI-SQUARE (χ²) TEST SONUÇLARI")
print("==============================\n")

for col in categorical_cols:

    print(f"\nKolon: {col}")

    # Frekans tabloları
    original_counts = data[col].value_counts()
    synthetic_counts = new_data[col].value_counts()

    # İki tabloyu hizalamak
    combined = pd.concat([original_counts, synthetic_counts], axis=1, sort=False)
    combined.columns = ["Original", "Synthetic"]
    combined = combined.fillna(0)

    # Chi-Square testi
    chi2, p, dof, expected = chi2_contingency(combined.T)

    print(f"Chi2 = {chi2:.3f}")
    print(f"p-value = {p:.4f}")

    if p > 0.05:
        print("✔ Dağılımlar BENZER — CTGAN bu kolonu doğru öğrenmiş.\n")
    else:
        print("✘ Dağılımlar farklı — CTGAN bu kolonu zayıf öğrenmiş.\n")

