# EnergyWise AI

**Smart Household Energy Monitoring & Prediction System Based on Machine Learning**

## 1. Masalah yang Diselesaikan

Penggunaan energi listrik rumah tangga sering kali sulit dipantau dan diperkirakan secara akurat. Pengguna belum tentu mengetahui pola konsumsi listrik, waktu penggunaan energi tertinggi, maupun perkiraan biaya listrik yang harus dibayarkan.

EnergyWise AI merupakan aplikasi berbasis web yang dikembangkan untuk membantu pengguna memantau konsumsi energi listrik, memperkirakan penggunaan listrik berdasarkan parameter kelistrikan, memprediksi konsumsi energi pada periode mendatang, serta menghitung estimasi biaya listrik.

Aplikasi ini menggunakan teknologi Machine Learning dan framework Django untuk menyediakan sistem monitoring dan prediksi konsumsi energi listrik.

## 2. Dataset

**Nama Dataset:** Household Power Consumption

**Sumber Dataset:** Kaggle – imtkaggleteam

Link: https://www.kaggle.com/datasets/imtkaggleteam/household-power-consumption

**Informasi Dataset:**
- Platform: Kaggle
- Nama pemilik dataset: imtkaggleteam
- Jumlah data yang digunakan: 1.048.575 baris (berdasarkan file CSV yang digunakan dalam project).
- Jumlah atribut: 9 kolom.
- Periode data: 16 Desember 2006 – 13 Desember 2008 (berdasarkan file yang digunakan).
- Interval pencatatan: Setiap 1 menit.
- Format file: CSV.

**Atribut Dataset:**

1. Date: Tanggal pencatatan.
2. Time: Waktu pencatatan.
3. Global_active_power: Daya aktif listrik rumah tangga (kW).
4. Global_reactive_power: Daya reaktif listrik (kW).
5. Voltage: Tegangan listrik (V).
6. Global_intensity: Intensitas arus listrik (A).
7. Sub_metering_1: Konsumsi listrik kelompok peralatan dapur (Wh).
8. Sub_metering_2: Konsumsi listrik kelompok peralatan laundry (Wh).
9. Sub_metering_3: Konsumsi listrik kelompok pemanas air dan pendingin ruangan (Wh).

Dataset digunakan sebagai sumber data historis untuk membangun model estimasi dan prediksi konsumsi energi listrik rumah tangga.

**Lisensi:** Mengikuti ketentuan lisensi yang tercantum pada halaman dataset Kaggle. Lisensi perlu diverifikasi sebelum pengumpulan project.

## 3. Metode

EnergyWise AI menggunakan pendekatan Machine Learning dengan metode regresi untuk mengestimasi dan memprediksi konsumsi energi listrik.

**A. Estimasi Konsumsi Listrik**

Sistem memperkirakan Global Active Power berdasarkan parameter kelistrikan, seperti Global Reactive Power, Voltage, dan Sub Metering.

**B. Prediksi Konsumsi Listrik Masa Depan**

Sistem memanfaatkan data historis konsumsi listrik, fitur waktu, serta nilai konsumsi pada periode sebelumnya untuk memperkirakan penggunaan listrik pada periode berikutnya.

**Algoritma yang digunakan:**

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

**Tahapan pengolahan data:**

- Data cleaning dan penanganan missing values.
- Feature engineering.
- Pembagian data training dan testing berdasarkan urutan waktu.
- Pelatihan dan perbandingan model.
- Evaluasi performa model.

## 4. Hasil dan Evaluasi

Evaluasi model dilakukan menggunakan tiga metrik:

- MAE (Mean Absolute Error).
- RMSE (Root Mean Squared Error).
- R² Score.

Hasil pengujian akan digunakan untuk membandingkan performa Linear Regression, Decision Tree Regressor, dan Random Forest Regressor.

**Tabel hasil evaluasi:**

| Algoritma | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | Menunggu pengujian | Menunggu pengujian | Menunggu pengujian |
| Decision Tree Regressor | Menunggu pengujian | Menunggu pengujian | Menunggu pengujian |
| Random Forest Regressor | Menunggu pengujian | Menunggu pengujian | Menunggu pengujian |

Model terbaik akan dipilih berdasarkan hasil evaluasi pada data testing.

## 5. Cara Menjalankan

**Persyaratan:**
- Python
- Django
- Library Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

**Langkah menjalankan aplikasi:**

1. Clone atau buka folder project EnergyWise AI.
2. Aktifkan virtual environment.

```powershell
.\venv\Scripts\Activate.ps1
```

3. Instal seluruh dependency.

```powershell
pip install -r requirements.txt
```

4. Jalankan migrasi database.

```powershell
python manage.py migrate
```

5. Jalankan server Django.

```powershell
python manage.py runserver
```

6. Buka aplikasi melalui browser:

```text
http://127.0.0.1:8000/
```

## 6. Fitur Aplikasi

EnergyWise AI memiliki fitur utama:

1. Dashboard Monitoring: Menampilkan ringkasan konsumsi energi listrik dan visualisasi data.
2. Energy Estimation: Memperkirakan konsumsi listrik berdasarkan parameter kelistrikan.
3. Energy Prediction: Memprediksi konsumsi listrik pada periode mendatang.
4. Cost Estimation: Menghitung estimasi biaya berdasarkan konsumsi energi dan tarif listrik.
5. Prediction History: Menyimpan dan menampilkan riwayat hasil prediksi.

## 7. Teknologi yang Digunakan

- Python: Bahasa pemrograman utama.
- Django: Framework pengembangan aplikasi web.
- HTML, CSS, JavaScript: Pengembangan antarmuka.
- Bootstrap 5: Framework desain responsif.
- Chart.js: Visualisasi grafik.
- Pandas dan NumPy: Pengolahan data.
- Scikit-learn: Pengembangan Machine Learning.
- SQLite: Penyimpanan data aplikasi.
- Joblib: Penyimpanan model Machine Learning.

## 8. Refleksi Singkat

Melalui pengembangan EnergyWise AI, proses pembelajaran Sistem Cerdas dapat diterapkan dalam permasalahan nyata, khususnya pemantauan dan prediksi konsumsi energi listrik rumah tangga.

Pengembangan aplikasi ini memberikan pengalaman dalam melakukan preprocessing data, membangun model Machine Learning, mengevaluasi performa algoritma, serta mengintegrasikan model ke dalam aplikasi web menggunakan Django.

Keterbatasan project ini adalah dataset berasal dari satu rumah tangga sehingga pola konsumsi listrik yang diperoleh belum tentu mewakili seluruh rumah tangga. Selain itu, kualitas prediksi dipengaruhi oleh kelengkapan dan karakteristik data historis.

Rencana pengembangan selanjutnya adalah meningkatkan kualitas model, memperluas variasi data, serta mengembangkan rekomendasi penghematan energi yang lebih informatif.