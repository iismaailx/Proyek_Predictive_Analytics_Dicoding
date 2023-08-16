# Proyek-Pertama-Predictive-Analytics-ML-Terapan-Dicoding
# Laporan Proyek Machine Learning Classification untuk Diagnosis Penyakit Stroke - Iis Ismail

## Domain Proyek

Stroke adalah kondisi medis yang serius terjadi ketika aliran darah ke bagian otak terganggu atau terhenti. Hal ini menyebabkan sel-sel otak mengalami kematian dalam periode singkat akibat kekurangan oksigen dan nutrisi. Kondisi ini dapat disebabkan oleh pembuluh darah otak yang tersumbat oleh gumpalan darah, mengakibatkan stroke iskemik, atau oleh pecahnya pembuluh darah yang menyebabkan perdarahan di otak, dikenal sebagai stroke hemoragik. Dalam proyek Machine Learning untuk diagnosis penyakit stroke, kami bertujuan untuk mengembangkan dan meningkatkan kemampuan klasifikasi untuk mengidentifikasi dan memprediksi risiko terjadinya stroke pada individu berdasarkan data medis dan faktor risiko yang relevan.

## Gejala Stroke:
Gejala stroke dapat bervariasi tergantung pada bagian otak yang terpengaruh. Beberapa gejala umum yang mungkin terjadi adalah:
1. Kelumpuhan atau Kelemahan: Terjadi pada salah satu sisi tubuh, misalnya, lengan atau kaki.
2. Gangguan Berbicara dan Memahami: Kesulitan berbicara, mencari kata, atau memahami percakapan.
3. Gangguan Penglihatan: Penglihatan kabur, ganda, atau hilang pada salah satu mata.
4. Kebingungan: Kesulitan memahami situasi atau lingkungan sekitar.
5. Pusing, Hilang Keseimbangan: Rasa pusing yang parah atau hilangnya keseimbangan.
6. Sakit Kepala Parah: Kadang-kadang disertai dengan muntah atau perubahan kesadaran.
7. Kelumpuhan Wajah: Salah satu sisi wajah mungkin turun atau terasa kesemutan.
## Faktor Risiko:
Beberapa faktor yang dapat meningkatkan risiko stroke meliputi:
1. Hipertensi (Tekanan Darah Tinggi): Tekanan darah tinggi dapat merusak pembuluh darah dan meningkatkan risiko penyumbatan atau pecahnya pembuluh darah otak.
2. Merokok: Merokok dapat merusak pembuluh darah dan meningkatkan risiko pembentukan gumpalan darah.
3. Diabetes: Diabetes dapat merusak pembuluh darah dan mengganggu sirkulasi darah.
3. Obesitas: Obesitas terkait dengan faktor risiko lain seperti tekanan darah tinggi, diabetes, dan penyakit jantung.
4. Riwayat Keluarga: Jika ada anggota keluarga yang pernah mengalami stroke, risiko Anda juga bisa lebih tinggi.


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
Data ini berisi 3419 pasien dengan riwayat riwayat penyakit yang berbeda dan usia serta dengan kebisaan meroko. Data ini memiliki beberapa fitur antara lain:
- id : number uniq pasien
- age : Usia anggota pasien
- gender : Jenis kelamin pasien
- hypertension : tekanan darah pasien
- heart_disease : penyakt jantung
- ever_married : staus pasien
- work_type : jenis pekerjaan pasien
- residence_type : jenis rumah pasien
- avg_glucose_level : tingkat glukosa pasien
- bmi : Body mass index. Perbandingan berat badan anggota dalam kg dan pangkat dua dari tinggi badan anggota dalam meter;
- smoking_status :kondisi seorang pasien dalam hal kebiasaan merokok tembakau
- stroke : kondisi seseoorang dikatakan stroke atau tidak dengan di gantikan oleh ( 1 jika kondisi stroke, 0 jika tidak)

## Data Preparation
Preparasi data dilakukan dengan tahapan sebagai berikut:
- menghapus data-data outliers 
  Dalam hal ini, akan dihapus data data outlier atau yang keluar dari trend. Pada proyek ini data outlier adalah data angka BMI dan avg_glucose_level yang berlebihan.
  ![outlier](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/outlier_age_bmi_glukosa.png)
- melihat kondisi data sampel

![gender](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/download.png)
![smoking_status](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/download%20lagi.png)
![ever_maried](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/download%20(1).png)
![residence_type](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/download%20(2)%5B.png)
![work_type](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/download%20work.png)

- melihat jumlah orang yang mengalami stroke dari semua fator
![plot_stroke](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/downloadstroke.png)
- mengobservasi korelasi antara fitur numerik dengan fitur target
![pairplot](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/pairplot.png)
bisa kita liat secara visual mengenai korelasi semua feature numerik terhadap target
-korelasi semua fitur numerik dengan correlation matrix
![correlationmat](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/corrmatrik.png)
- melihat distribusi umur terhadap stroke
  ![agestroke](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/dist_agestrok.png)
- melihat distribusi bmi terhadapt stroke
   ![bmistroke](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/dist_bmi_strok.png)
- melihat distribusi data 0 1 di dalam data taget stroke itu sendiri
- ![strokestroke](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/Screenshot%202023-08-16%20140159.png)
Setelah kita lihat beberapa persebaran target dan bmi terhadap target stroke kurang terdistribusi dengan baik begitupun ketika kita melihat pesebaran data stroke sendiri terjadi imbalance, seperti berikut : 
![imbalancestroke](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/Screenshot%202023-08-16%20140205.png)
agar data menjadi balance kita perlu melakukan resample data seperti code di bawah ini :
![fiximbalance](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/Screenshot%202023-08-16%20140226.png)
setelah kita melalukan resample, maka data akan menjadi balance seperti gambar di bawah berikut :
![fiximbalance](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/Screenshot%202023-08-16%20140236.png)
## Modeling
- Modeling proyek ini dilakukan dengan 3 metode yang akan dibandingkan satu sama lain antara lain KNN, random forest, dan SVM. Parameter K pada KNN yang digunakan adalah sebesar 3, lalu parameter RF yang digunakan adalah seperti n_estimators=150,criterion='entropy',random_state = 123 dan untuk SVM kita biarkan secara defualt saja.
- Pada proyek ini, MSE terkecil dicetak oleh RF, lalu disusul oleh KNN, dan MSE terbesar adalah model SVM.
![MSE](https://github.com/iismaailx/Proyek_Predictive_Analytics_Dicoding/blob/main/Gambar/mse.png)
- berikut hasil akurasinya : 
| Model | KKN      | RF      | SVM      |   |
|-------|----------|---------|----------|---|
| ACC   | 0.975336 | 0.99701 | 0.908819 |   |
|       |          |         |          |   |
|       |          |         |          |   |

Dalam proyek ini, saya akan menggunakan RF karena nilai akurasinya jauh lebih tinggi.

## Evaluation
- Dibutuhkan optimalisasi lebih lanjut utamanya pada KNN dan SVM agar bisa mendapatkan hasil prediksi yang lebih mendekati hasil aslinya.

**---Ini adalah bagian akhir laporan---**

