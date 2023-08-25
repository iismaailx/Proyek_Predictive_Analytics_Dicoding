# Proyek-Pertama-Predictive-Analytics-ML-Terapan-Dicoding
# Laporan Proyek Machine Learning Classification untuk Diagnosis Penyakit Stroke - Iis Ismail

## Domain Proyek
Stroke adalah salah satu penyakit yang memiliki dampak signifikan terhadap kesehatan masyarakat dan tingkat kematian yang tinggi di seluruh dunia. Menurut Organisasi Kesehatan Dunia (WHO), stroke merupakan penyebab kematian nomor dua di dunia, dengan lebih dari 11 juta orang mengalami stroke setiap tahunnya. Di samping angka kematian, banyak penderita stroke yang mengalami cacat jangka panjang, seperti gangguan berbicara, gerakan terbatas, dan masalah kognitif.
Penyebab stroke dapat beragam, tetapi sebagian besar terjadi akibat pembuluh darah yang menyuplai darah ke otak tersumbat oleh gumpalan darah atau pecahnya pembuluh darah. Stroke iskemik (akibat penyumbatan pembuluh darah) dan stroke hemoragik (akibat pecahnya pembuluh darah) adalah dua jenis utama stroke. Deteksi dan diagnosa dini terkait risiko stroke menjadi sangat penting, karena tindakan pencegahan yang tepat waktu dapat mengurangi dampak yang ditimbulkan.
Dalam upaya untuk meningkatkan deteksi dini dan diagnosis risiko stroke, penggunaan teknologi Machine Learning (ML) telah menjadi area penelitian yang semakin menarik. Dengan memanfaatkan data medis dan faktor risiko yang relevan, seperti riwayat medis, usia, tekanan darah, kadar gula darah, dan faktor gaya hidup, model ML dapat mengklasifikasikan individu berdasarkan risiko mereka mengalami stroke. Proyek ini bertujuan untuk mengembangkan dan meningkatkan kemampuan model klasifikasi untuk mengidentifikasi dan memprediksi risiko terjadinya stroke pada individu.
Penggunaan ML dalam diagnosis penyakit stroke dapat membantu tenaga medis dalam memberikan tindakan pencegahan yang lebih efektif dan personal pada individu yang berisiko. Dengan mengidentifikasi faktor risiko yang berkaitan dengan stroke, model ML dapat memberikan informasi yang lebih baik dalam membuat keputusan klinis dan mengarahkan perawatan pasien.

Referensi:
1. World Health Organization (WHO). (2021). The top 10 causes of death. [https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death]
2. Benjamin, E. J., Muntner, P., Alonso, A., Bittencourt, M. S., Callaway, C. W., Carson, A. P., ... & Virani, S. S. (2019). Heart disease and stroke statistics—2019 update: a report from the American Heart Association. Circulation, 139(10), e56-e528.
3. Nannoni, S., Del Bene, A., Palumbo, V., Petrone, L., Nesi, M., Pescini, F., ... & Inzitari, D. (2018). Impact of atrial fibrillation on the decision-making process in acute ischemic stroke: insights into “real-world” clinical practice from the Italian ACute Stroke Registry. Journal of the Neurological Sciences, 387, 115-121.

### Gejala Stroke:
Gejala stroke dapat bervariasi tergantung pada bagian otak yang terpengaruh. Beberapa gejala umum yang mungkin terjadi adalah:
| No 	| Gejala 	| Keterangan 	|
|---	|:---:	|:---:	|
| 1 	| Kelumpuhan atau Kelemahan 	| Terjadi pada salah satu sisi tubuh, misalnya, lengan atau kaki. 	|
| 2 	| Gangguan Berbicara dan Memahami 	| Kesulitan berbicara, mencari kata, atau memahami percakapan. 	|
| 3 	| Gangguan Penglihatan 	| Penglihatan kabur, ganda, atau hilang pada salah satu mata. 	|
| 4 	| Kebingunan 	| Kesulitan memahami situasi atau lingkungan sekitar. 	|
| 5 	| Pusing, hilang keseimbangan 	| Rasa pusing yang parah atau hilangnya keseimbangan. 	|
| 6 	| Sakit Kepala Parah 	| Kadang-kadang disertai dengan muntah atau perubahan kesadaran. 	|
| 7 	| Kelumpuhan Wajah 	| Salah satu sisi wajah mungkin turun atau terasa kesemutan. 	|
### Faktor Risiko:
Beberapa faktor yang dapat meningkatkan risiko stroke meliputi:
| No 	| Faktor 	| Keterangan 	|
|---	|:---:	|:---:	|
| 1 	| Hipertensi (Tekanan Darah Tinggi) 	| Tekanan darah tinggi dapat merusak pembuluh darah dan meningkatkan risiko penyumbatan atau pecahnya pembuluh darah otak. 	|
| 2 	| Merokok 	| Merokok dapat merusak pembuluh darah dan meningkatkan risiko pembentukan gumpalan darah 	|
| 3 	| Diabetes 	| Diabetes dapat merusak pembuluh darah dan mengganggu sirkulasi darah 	|
| 4 	| Obesitas 	| Obesitas terkait dengan faktor risiko lain seperti tekanan darah tinggi, diabetes, dan penyakit jantung. 	|
| 5 	| Riwayat Keluarga 	| Jika ada anggota keluarga yang pernah mengalami stroke, risiko Anda juga bisa lebih tinggi 	|


## Business Understanding
Dataset yang digunakan pada proyek ini didapatkan dari link di bawah ini:
[https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset](https://raw.githubusercontent.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/main/healthcare-dataset-stroke-data.csv)

### Problem Statements

Dalam konteks proyek ini, terdapat tantangan yang berkaitan dengan pengembangan model Machine Learning untuk prediksi risiko stroke berdasarkan data medis dan faktor risiko. Beberapa masalah yang muncul antara lain:

1. Identifikasi Faktor Risiko Stroke:
Diperlukan pendekatan yang mampu mengidentifikasi faktor-faktor yang signifikan dalam meningkatkan risiko terjadinya stroke pada individu. Bagaimana cara menggabungkan dan menganalisis data medis serta informasi faktor risiko yang beragam sehingga dapat mengidentifikasi faktor-faktor utama yang berkaitan dengan stroke?

2. Pemilihan Model dengan Performa Terbaik:
Tantangan lain adalah menentukan model Machine Learning yang memiliki performa paling baik dalam mengklasifikasikan individu berdasarkan risiko stroke. Bagaimana cara memilih di antara berbagai algoritma, seperti Random Forest, Support Vector Machine, atau algoritma lainnya, agar dapat memberikan hasil prediksi yang akurat dan andal?

3. Evaluasi Akurasi dan Error:
Dalam pengembangan model prediksi risiko stroke, penting untuk mengevaluasi sejauh mana akurasi model dalam mengklasifikasikan individu yang berisiko dan tidak berisiko mengalami stroke. Bagaimana mengukur akurasi dan bagaimana memperkirakan tingkat error yang mungkin terjadi pada hasil prediksi model terbaik?

### Goals

1. Mengembangkan model Machine Learning yang mampu mengidentifikasi faktor-faktor risiko yang berkaitan dengan stroke berdasarkan data medis dan informasi faktor risiko.
2. Menentukan model dengan performa terbaik untuk memprediksi risiko stroke pada individu.
3. Mengukur akurasi dan memperkirakan tingkat error pada hasil prediksi model terbaik.

### Solution statements
Dalam upaya mencapai tujuan proyek, berikut adalah beberapa solusi yang akan diimplementasikan:
1. Pengembangan Model Klasifikasi:
Pertama, akan diimplementasikan beberapa model klasifikasi seperti Random Forest, Support Vector Machine, dan algoritma lainnya. Model ini akan dilatih dengan menggunakan dataset yang mencakup data medis dan faktor risiko individu. Dengan melibatkan berbagai faktor, model dapat mengidentifikasi pola-pola yang berkaitan dengan risiko stroke.
2. Penyetelan Hyperparameter:
Dilakukan eksplorasi dan penyetelan hyperparameter pada model klasifikasi untuk meningkatkan performa. Penggunaan algoritma dengan konfigurasi optimal dapat menghasilkan hasil prediksi yang lebih akurat. Metrik evaluasi seperti akurasi, presisi, recall, dan F1-score akan digunakan untuk mengukur peningkatan performa model.
3. Validasi dan Evaluasi:
Model yang telah dikembangkan akan divalidasi menggunakan metode validasi silang dan pengujian pada data yang belum pernah dilihat sebelumnya. Hal ini akan memberikan gambaran yang lebih baik tentang seberapa baik model dapat melakukan prediksi pada situasi dunia nyata. Selain itu, akan dilakukan estimasi tingkat error melalui pengujian pada dataset uji yang berbeda.

## Data Understanding
Data ini berisi 3419 pasien dengan riwayat riwayat penyakit yang berbeda dan usia serta dengan kebisaan meroko. Data ini memiliki beberapa Attribute antara lain sebagai berikut :
| Attribute 	| Keterangan 	|
|---	|---	|
| id 	| Number unix pasien 	|
| Age 	| Umur pasien 	|
| Gender 	| Jenis kelamin pasien 	|
| Hypertension 	| Tekanan darah pasien 	|
| Heart_disease 	| Penyakit jantung 	|
| Ever_married 	| Status pernikahan pasien 	|
| Work_type 	| Jenis pekerjaan pasien 	|
| Residence_type 	| Jenis rumah pasien 	|
| Avg_glucose_level 	| Tingkat glukosa pasien 	|
| BMI 	| Body mass index. Perbandingan berat badan anggota dalam kg dan pangkat dua dari tinggi badan anggota dalam meter 	|
| Smoking_status 	| kondisi seorang pasien dalam hal kebiasaan merokok tembakau 	|
| Stroke 	| Kondisi seseoorang dikatakan stroke atau tidak dengan di gantikan oleh ( 1 jika kondisi stroke, 0 jika tidak) 	|

## Data Preparation
Preparasi data dilakukan dengan tahapan sebagai berikut:
- Gunakan method describe untuk menghasilkan ringkasan statistik dari kolom-kolom dalam DataFrame. Ini memberikan informasi statistik penting seperti rata-rata, standar deviasi, nilai minimum, kuartil, dan lain-lain tentang data dalam kolom-kolom DataFrame.

**index**|**id**|**age**|**hypertension**|**heart\_disease**|**avg\_glucose\_level**|**bmi**|**stroke**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
count|5110.0|5110.0|5110.0|5110.0|5110.0|4909.0|5110.0
mean|36517.82935420744|43.226614481409|0.0974559686888454|0.05401174168297456|106.1476771037182|28.893236911794666|0.0487279843444227
std|21161.721624827165|22.61264672311349|0.29660667423379117|0.22606298750336543|45.28356015058198|7.854066729680164|0.2153198569802376
min|67.0|0.08|0.0|0.0|55.12|10.3|0.0
25%|17741.25|25.0|0.0|0.0|77.245|23.5|0.0
50%|36932.0|45.0|0.0|0.0|91.88499999999999|28.1|0.0
75%|54682.0|61.0|0.0|0.0|114.09|33.1|0.0
max|72940.0|82.0|1.0|1.0|271.74|97.6|1.0

-Fungsi info() pada pandas digunakan untuk mendapatkan informasi rinci tentang DataFrame, termasuk tipe data kolom, jumlah entri non-null, penggunaan memori, dan lain-lain.


**#**|**Column**|**Non-Null Count**|**Dtype**
:-----:|:-----:|:-----:|:-----:
0|id|5110 non-null|int64
1|gender|5110 non-null|object
2|age|5110 non-null|float64
3|hypertension|5110 non-null|int64
4|heart\_disease|5110 non-null|int64
5|ever\_married|5110 non-null|object
6|work\_type|5110 non-null|object
7|Residence\_type|5110 non-null|object
8|avg\_glucose\_level|5110 non-null|float64
9|bmi|4909 non-null|float64
10|smoking\_status|5110 non-null|object
11|stroke|5110 non-null|int64
- Menghapus kolom yang tidak di butuhkan seperti kolom id


**gender**|**age**|**hypertension**|**heart\_disease**|**ever\_married**|**work\_type**|**Residence\_type**|**avg\_glucose\_level**|**BMI**|**smoking\_status**|**stroke**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
|Male|67.0|0|1|Yes|Private|Urban|228.69|36.6|formerly smoked
|Female|61.0|0|0|Yes|Self-employed|Rural|202.21|NaN|never smoked
|Male|80.0|0|1|Yes|Private|Rural|105.92|32.5|never smoked
|Female|49.0|0|0|Yes|Private|Urban|171.23|34.4|smokes
|Female|79.0|1|0|Yes|Self-employed|Rural|174.12|24.0|never smoked

-Mengecek data yang kosong atau bernilai null pada dataset
**#**|**Column**|**Non-Null Count**|**Dtype**
:-----:|:-----:|:-----:|:-----:
0|gender|4909 non-null|object
1|age|4909 non-null|float64
2|hypertension|4909 non-null|int64
3|heart\_disease|4909 non-null|int64
4|ever\_married|4909 non-null|object
5|work\_type|4909 non-null|object
6|Residence\_type|4909 non-null|object
7|avg\_glucose\_level|4909 non-null|float64
8|bmi|4909 non-null|float64
9|smoking\_status|4909 non-null|object
10|stroke|4909 non-null|int64
- Menghapus data yang null menggunakan kode berikut :
```
stroke_df.dropna(inplace=True)
stroke_df.isna().sum()
```
dengan menjalankan kode berikut maka data yang null akan di hapus, lalu cek kemabli datanya

| gender 	| 0 	|
|---	|---	|
| age 	|  0	|
| hypertension 	|  	0|
| heart_disease 	| 0 	|
| ever_married 	|  0	|
| work_type 	|  0	|
| Residence_type 	|  0	|
| avg_glucose_level 	|  0	|
| bmi 	| 0 	|
| smoking_status 	|  0	|
| stroke 	|  0	|

- Membuat fungsi untuk menghandle data-data outliers
```python
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

index_list = sorted(set(index_list))
```

Dalam hal ini, akan dihapus data data outlier atau yang keluar dari trend. Pada proyek ini data outlier 
berasal dari data BMI dan avg_glucose_level yang berlebihan.
  
- Melihat kondisi data sampel dan shape ketika sudah dibersihkan data outliernya dengan kode berikut:

```
  shape_sebelumnya = stroke_df.shape
  stroke_df = stroke_df.drop(index_list)
  shape_sesudahnya = stroke_df.shape
  print(f'Shape data sebelumnya : {shape_sebelumnya}')
  print(f'Shape data sebelumnya : {shape_sesudahnya}')
```
  Shape yang di hasilkan sebagai berikut:
  Shape data sebelumnya : (4909, 11)
  Shape data sebelumnya : (3419, 11)
- Menampilkan Visualisasi plot bar Gender untuk melihat persebaran datanya dengan kode berikut:
  
```
  feature = categorical_features[0]
  count = stroke_df[feature].value_counts()
  percent = 100*stroke_df[feature].value_counts(normalize=True)
  df = pd.DataFrame({'Jumlah sampel': count, 'Persentase':percent.round(1)})
  print(df)
  count.plot(kind='bar', title=feature, color="#6EC4D4")
```
Menghasilkan output sebagai berikut:
| Gender 	| Jumlah Sample 	| Persentase 	|
|---	|---	|---	|
| Male 	| 2007 	| 58.7 	|
| Female 	| 1412 	| 41.3 	|

- Selanjutnya menampilkan Visualisasi plot bar untuk melihat persebaran data feature categorical terhadapat data kolom target yaitu stroke dengan kode berikut:

```
plt.figure(figsize=(15, 10))
for i in range(len(categorical_features)):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=stroke_df[categorical_features[i]], hue = stroke_df['stroke'], palette='bone')
```

dikarenakan di markdown tidak bisa menampilkan gambar, maka penjelasan tahapan data preparation akan di lampirkan pada link jupyternotebook.
[Link_Jupyternotebook]([https://colab.research.google.com/drive/1kEiCKvvBSXgfosjfWBjLT-WDFVgOf0uc?authuser=1#scrollTo=ZTFszJx4qfYD])

## Modeling
- Modeling proyek ini dilakukan dengan 3 metode yang akan dibandingkan satu sama lain antara lain KNN, random forest, dan SVM. Parameter K pada KNN yang digunakan adalah sebesar 3, lalu parameter RF yang digunakan adalah seperti n_estimators=150,criterion='entropy',random_state = 123 dan untuk SVM kita biarkan secara defualt saja.
- Pada proyek ini, MSE terkecil dicetak oleh RF, lalu disusul oleh KNN, dan MSE terbesar adalah model SVM.
  
| Model 	| KKN 	| RF 	| SVM 	|
|---	|---	|---	|---	|
| ACC 	| 0.975336 	| 0.99701 	| 0.908819 	|

Dalam proyek ini, baiknya menggunakan RF karena nilai akurasinya jauh lebih tinggi.

## Evaluation
- Dibutuhkan optimalisasi lebih lanjut utamanya pada KNN dan SVM agar bisa mendapatkan hasil prediksi yang lebih mendekati hasil aslinya.

**---Ini adalah bagian akhir laporan---**

