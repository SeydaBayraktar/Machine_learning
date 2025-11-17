import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def check_df(dataframe, head=5):
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.head(head))
    print(dataframe.tail(head))
    print(dataframe.isnull().sum())
    # Sayısal değişkenlerin dağılım bilgisi
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
data = pd.read_csv("C:/Users/ŞEYDA/Desktop/dataset.csv", sep=';', decimal=',')
df = data
check_df(df)

#2. Kategorik Değişken Analizi (Analysis of Categorical Variables)

def unique_value_counts(dataframe):
    for column in dataframe.columns:
        print(dataframe[column].value_counts())

# fonksiyonu kullanarak DataFrame'deki tüm sütunların benzersiz değerlerini gösterelim
unique_value_counts(df)

#kategorik değişkenlerimizi cat_cols variable'ına atayalım.
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

#numerik ama kategorik değişkenlerimizi bulalım.
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_cols = cat_cols + num_but_cat

#kategorik olup kardinalitesi yüksek değişkenlerimizi bulalım. Bu örnekte 20 olsun. 

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = [col for col in cat_cols if col not in cat_but_car]
#cat_cols içindeki değişkenler, yüksek kardinaliteye sahip olanlar 
#hariç olacak şekilde filtrelenir. 

num_cols = [col for col in df.columns if df[col].dtype in ["int", "float"]]

# num_cols'un içinde olup, cat_cols'da olmayan ( cat_cols => kategorik )
# cat_cols'u Kategorik Değişken Analizi'nde belirledik.

num_cols = [col for col in num_cols if col not in cat_cols]
print(cat_cols)
print(cat_but_car)
print(num_cols)
print(num_but_cat)

# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# kategorik ve target ilişkisi
df["Inspection results"] = df["Inspection results"].map({"Fail": 1, "Pending": 0, "Pass": 0})
df.groupby("Product type")["Inspection results"].mean()

# fonksiyon

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "Inspection results", "Product type")  # 
target_summary_with_cat(df, "Inspection results", "Supplier name")  # 
target_summary_with_cat(df, "Inspection results", "Customer demographics") 

# sayısalla target ilişkisi

df.groupby("Inspection results").agg({"Defect rates": "mean"})

# fonksiyon

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "Inspection results", "Defect rates")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/ŞEYDA/Desktop/dataset.csv", sep=';', decimal=',')
df = df.iloc[:, 1:-1]
df.head()

# önce sayısal değerler.
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
# Hedef değişkeni sayısala çevir
df["Inspection results"] = df["Inspection results"].map({"Fail": 1, "Pending": 0, "Pass": 0})
# Hedef değişken
if "Inspection results" not in num_cols:
    num_cols.append("Inspection results")

# korelasyon hesaplamak için => corr() metodunu kullanırız.
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()
