# Proyek-Pertama-Predictive-Analytics-ML-Terapan-Dicoding
# Laporan Proyek Machine Learning Classification untuk Diagnosis Penyakit Stroke - Iis Ismail

## Domain Proyek

Stroke adalah kondisi medis yang serius terjadi ketika aliran darah ke bagian otak terganggu atau terhenti. Hal ini menyebabkan sel-sel otak mengalami kematian dalam periode singkat akibat kekurangan oksigen dan nutrisi. Kondisi ini dapat disebabkan oleh pembuluh darah otak yang tersumbat oleh gumpalan darah, mengakibatkan stroke iskemik, atau oleh pecahnya pembuluh darah yang menyebabkan perdarahan di otak, dikenal sebagai stroke hemoragik. Dalam proyek Machine Learning untuk diagnosis penyakit stroke, kami bertujuan untuk mengembangkan dan meningkatkan kemampuan klasifikasi untuk mengidentifikasi dan memprediksi risiko terjadinya stroke pada individu berdasarkan data medis dan faktor risiko yang relevan.

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

Masalah pada proyek ini antara lain:
- Apa saja faktor-faktor apa saja yang dappat mengakibatkan seseorang terkena stroke?
- Apa model yang paling akurat pada kasus proyek ini?
- Berapa error yang akan terjadi pada hasil prediksi model terakurat pada kasus proyek ini?


### Goals

Tujuan dari proyek ini antara lain:
- Menentukan faktor-faktor apa saja yang mempengaruhi seseorang dapat terkena stroke.
- Menentukan model yang bisa memprediksi paling akurat pada proyek ini.
- Memprediksi dan mendiagnosa seseorang bisa terkena stroke.

### Solution statements
- Solusi ini akan membandingkan akurasi dari algortima K nearest Neighbour, Random forest, dan support vector machine 


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
0|Male|67.0|0|1|Yes|Private|Urban|228.69|36.6|formerly smoked
1|Female|61.0|0|0|Yes|Self-employed|Rural|202.21|NaN|never smoked
2|Male|80.0|0|1|Yes|Private|Rural|105.92|32.5|never smoked
3|Female|49.0|0|0|Yes|Private|Urban|171.23|34.4|smokes
4|Female|79.0|1|0|Yes|Self-employed|Rural|174.12|24.0|never smoked

-Mengecek data yang kosong atau bernilai null pada dataset
gender|0
::|::
age|0
hypertension|0
heart\_disease|0
ever\_married|0
work\_type|0
Residence\_type|0
avg\_glucose\_level|0
bmi|201
smoking\_status|0
stroke|0
- Menghapus data yang kosong atau null
  gender|0
::|::
age|0
hypertension|0
heart\_disease|0
ever\_married|0
work\_type|0
Residence\_type|0
avg\_glucose\_level|0
bmi|0
smoking\_status|0
stroke|0

- menghapus data-data outliers 
  Dalam hal ini, akan dihapus data data outlier atau yang keluar dari trend. Pada proyek ini data outlier 
  adalah data angka BMI dan avg_glucose_level yang berlebihan.
- melihat kondisi data sampel
- melihat jumlah orang yang mengalami stroke dari semua faktor
- mengobservasi korelasi antara fitur numerik dengan fitur target
- korelasi semua fitur numerik dengan correlation matrix
- melihat distribusi umur terhadap stroke
- melihat distribusi BMI terhadapt stroke
- melihat distribusi data 0 1 di dalam data taget stroke itu sendiri
Setelah melihat beberapa persebaran target dan bmi terhadap target stroke kurang terdistribusi dengan baik begitupun ketika kita melihat persebaran data stroke sendiri terjadi imbalance. 

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

