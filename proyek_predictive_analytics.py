# -*- coding: utf-8 -*-
"""Proyek Predictive Analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kEiCKvvBSXgfosjfWBjLT-WDFVgOf0uc
"""

# Nama Ismai

# langkah pertama yaitu kita perlu mengimport library yang kita perlukan terlebih dahulu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn .ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

file_path = "https://raw.githubusercontent.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/main/healthcare-dataset-stroke-data.csv"
stroke_df = pd.read_csv(file_path)
stroke_df.info()

stroke_df.describe(include='all')

# menghapus kolom yang tidak di perlukan
stroke_df.drop(['id'], axis=1, inplace=True)
print('done')

stroke_df.head(5)

#mengecek adanya duplikat pada data
stroke_df.duplicated().sum()

#mengecek data yang kosong atau null
stroke_df.isna().sum()

# menghapus data yang nul, karena kita tidak dapat mengisi data bmi sembarangan takutnya malah membuat model kita tidak valid
stroke_df.dropna(inplace=True)
stroke_df.isna().sum()

# Melihat Outlier pada data kita
name_col = ['age', 'bmi', 'avg_glucose_level']
plt.figure(figsize=(12, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x=stroke_df[name_col[i]], color='blue')
    plt.title(name_col[i])
plt.show()

# membuat fungsi untuk handle outlier
def handling_outliers(data, column):
    Q1 = stroke_df[column].quantile(.25)
    Q3 = stroke_df[column].quantile(.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5*IQR)
    upper_bound = Q1 + (1.5*IQR)
    result = stroke_df.index[(stroke_df[column]< lower_bound) | (stroke_df[column] > upper_bound)]
    return result

data_columns = ['age', 'bmi', 'avg_glucose_level']
index_list = []
for column in data_columns:
    index_list.extend(handling_outliers(stroke_df, column))

# menghapus duplicated indices in the ondex_list and sort it
index_list = sorted(set(index_list))

# mengecek shape dataset
shape_sebelumnya = stroke_df.shape
stroke_df = stroke_df.drop(index_list)
shape_sesudahnya = stroke_df.shape
print(f'Shape data sebelumnya : {shape_sebelumnya}')
print(f'Shape data sebelumnya : {shape_sesudahnya}')

# cek kembali outlier pada data
sns.boxplot(x=stroke_df['age'])

"""# **Univariate Analysis**
Selanjutnya, kita akan melakukan proses analisis data dengan teknik Univariate EDA. Pertama, Anda bagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features.
"""

stroke_df.info()

# membuat list feature dari dataset kita
numerical_features =['age', 'avg_glucose_level', 'bmi']
categorical_features=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease']

feature = categorical_features[0]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'Jumlah sampel': count, 'Persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature, color="#6EC4D4")

feature = categorical_features[1]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'Jumlah sampel': count, 'Persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature, color="#6EC4D4")

feature = categorical_features[2]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'Jumlah sampel': count, 'Persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature, color="#6EC4D4")

feature = categorical_features[3]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'Jumlah sampel': count, 'Persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature, color="#6EC4D4")

feature = categorical_features[4]
count = stroke_df[feature].value_counts()
percent = 100*stroke_df[feature].value_counts(normalize=True)
df = pd.DataFrame({'Jumlah sampel': count, 'Persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature, color="#6EC4D4")

plt.figure(figsize=(15, 10))
for i in range(len(categorical_features)):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=stroke_df[categorical_features[i]], hue = stroke_df['stroke'], palette='bone')

"""## **Numerical Features**
Untuk mengamati hubungan antara fitur numerik, kita akan menggunakan fungsi pairplot(). Kita juga akan mengobservasi korelasi antara fitur numerik dengan fitur target menggunakan fungsi corr().
"""

# melihat hubungan korelasi setiap data numerik dengan pairplot
sns.pairplot(stroke_df, diag_kind = 'kde')

# mencari korelasi antar variabel angka
plt.figure(figsize=(10, 8))
correlation_matrix = stroke_df.corr().round(2)
# cetak nilai korelasi dalam heatmap
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5)
plt.title('Correlation Matrix untuk fitur numerik', size=20)

# melihat distribusi  data umur terhadap data stroke
sns.displot(data= stroke_df, x='age', hue ='stroke', kind='kde', palette = 'bone', height=4.5)
plt.show()

# melihat distribusi  data bmi terhadap data stroke
sns.displot(data=stroke_df, x='bmi', hue='stroke', kind='kde', palette='bone', height=4.5)
plt.show()

# melihat persebaran data stroke sendiri, apakah imbalance atau balance. karena ini mempunyai pengruh yang besar terhadap model yang akan di buat nantinya
stroke_dt = dict(stroke_df['stroke'].value_counts())
fig = px.pie(names = stroke_dt.keys(), values = stroke_dt.values(), title='Data Stroke', color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_traces(textposition='inside', textinfo='percent+label')

"""setelah kita melihat mengenai oposisi pada data, ternyata data stroke kita tidak seimbang. mari kita benarkan"""

# tahap ini disebut dengan tahap Data Preprocessing
# kita perlu resamble data yang tidak seimbang terlebih dahulu
stroke_0 = stroke_df[stroke_df.iloc[:, -1] == 0]
stroke_1 = stroke_df[stroke_df.iloc[:, -1] == 1]
stroke_df['stroke'].value_counts()

from sklearn.utils import resample
stroke_1 = resample(stroke_1, replace=True, n_samples=stroke_0.shape[0], random_state=123)

stroke_df = np.concatenate((stroke_0, stroke_1))
stroke_dtf = pd.DataFrame(stroke_df)
stroke_dtf.columns =['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type',
                     'Residence_type', 'avg_glucose_level', 'bmi','smoking_status', 'stroke']

# kita cek apakah data sudah balance atau belum
stroke_ = dict(stroke_dtf['stroke'].value_counts())
fig = px.pie(names = ['False', 'True'], values = stroke_.values(), title='Stroke Data', color_discrete_sequence=px.colors.sequential.Aggrnyl)
fig.update_traces(textposition='inside', textinfo='percent+label')

"""bisa kita lihat datanya sekarang menjadi seimbang"""

# karena kita mempunyai tipe data categorical, kita perlu melalukan oneshot encoding!
stroke_dtf = pd.get_dummies(data=stroke_dtf, columns = ['gender','ever_married','work_type',
                                                        'Residence_type','smoking_status'] ,drop_first=True)

stroke_dtf.head(5)

# split data feature dan target
x = stroke_dtf.drop('stroke', axis =1)
y = pd.to_numeric(stroke_dtf['stroke'])

# lakukan proses data scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =.20)

#Melihat jumlah data, data latih, dan data uji
print(f'Total # of sample in whole dataset: {len(x)}')
print(f'Total # of sample in train dataset: {len(x_train)}')
print(f'Total # of sample in test dataset: {len(x_test)}')

# buat prediksi KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = accuracy_score(y_test, y_pred)
acc_knn

# buat model prediksi Random Forest
RF =  RandomForestClassifier(n_estimators=150,criterion='entropy',random_state = 123)
RF.fit(x_train, y_train)
y_pred = RF.predict(x_test)
acc_rf = accuracy_score(y_test,y_pred)
acc_rf

# buat model prediksi menggunakan SVM
SVM = SVC()
SVM.fit(x_train, y_train)
y_pred = SVM.predict(x_test)
acc_svm = accuracy_score(y_test, y_pred)
acc_svm

# melihat perbandingan akurasi pada setiap model dengan terlebih dahulu meembuatnya sebagai dataFrame
best_model = {
    'KNN' : [acc_knn],
    'Random_Forest' : [acc_rf],
    'SVM': [acc_svm]
}
acc_modeldf = pd.DataFrame(best_model)
acc_modeldf

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','SVM'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'SVM': SVM}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""jika dilihat model dapat memprediksi dengan baik, namun alangkah baiknya kita menggunaskan model dengan akurasinya yang paling tinggi"""