# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech (Jaya Jaya Institute)

## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik.

### Permasalahan Bisnis
Berikut adalah permasalahan bisnis yang dihadapi oleh Jaya jaya institute
- Jumlah mahasiswa dropout yang tinggi (1.421 dari 4.424 mahasiswa).
- Belum adanya sistem prediksi untuk deteksi dini mahasiswa yang berpotensi dropout.
- Kebutuhan akan dashboard visual untuk membantu monitoring performa dan pengambilan keputusan berbasis data.


### Cakupan Proyek
1. Melakukan analisis deskriptif terhadap data mahasiswa berdasarkan status akademik, pembayaran, beasiswa, dan jurusan.
2. Membangun model machine learning untuk memprediksi kemungkinan mahasiswa melakukan dropout.
3. Mengembangkan dashboard visual interaktif menggunakan Streamlit.
4. Memberikan rekomendasi strategis untuk menurunkan angka dropout.

### Persiapan

Sumber data: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Setup environment:
1. Membuat environment baru bernama newenv
```
python -m venv newenv
```
2. Aktivasi environment
```
.\newenv\Scripts\activate
```
3. Menginstal package yang dibutuhkan
```
pip install -r requirements.txt
```

## Business Dashboard
link dashboard: https://lookerstudio.google.com/u/0/reporting/99133267-df9f-422b-bff7-33fa264960a8/page/BYBKF

<img src="image\mosaicnim-dashboard.jpg" alt="alt text" width="whatever" height="whatever">

Berikut adalah penjelasan masing-masing visualisasi yang terdapat dalam dashboard:
1. **Distribusi Status Mahasiswa**
   - Menampilkan jumlah dan proporsi mahasiswa dalam kategori: Graduate, Dropout, dan Enrolled.
   - Insight: 49.9% lulus, 32.1% dropout, dan 17.9% masih aktif.

2. **Distribusi Berdasarkan Gender**
   - Menunjukkan perbandingan status mahasiswa berdasarkan jenis kelamin.
   - Temuan: Mahasiswa laki-laki memiliki persentase dropout lebih tinggi dibanding perempuan.

3. **Hubungan Pembayaran Tepat Waktu dengan Status**
   - Menganalisis dampak ketepatan pembayaran terhadap kelulusan.
   - Mahasiswa yang membayar tepat waktu lebih banyak yang lulus dibanding yang dropout.

4. **Pengaruh Beasiswa terhadap Status**
   - Membandingkan status mahasiswa yang menerima beasiswa dengan yang tidak.
   - Sebagian besar mahasiswa yang tidak menerima beasiswa cenderung dropout atau lulus, sementara yang menerima beasiswa lebih banyak yang lulus dan jumlah dropout-nya jauh lebih sedikit. Ini mengindikasikan bahwa pemberian beasiswa mungkin berkontribusi positif terhadap kelulusan mahasiswa dan menurunkan angka dropout.

5. **Pengaruh Tunggakan (Hutang) terhadap Status**
   - Menganalisis hubungan antara tunggakan(debitur) dan status mahasiswa.
   - Mayoritas mahasiswa yang bukan debitur memiliki angka kelulusan yang sangat tinggi (2108), sedangkan mahasiswa yang merupakan debitur cenderung memiliki angka dropout yang lebih tinggi (312) dibandingkan yang lulus (101). Hal ini mengindikasikan bahwa memiliki utang mungkin berdampak negatif terhadap keberhasilan studi mahasiswa

6. **Distribusi Status Mahasiswa Berdasarkan Jurusan**
   - Memetakan jurusan-jurusan dengan tingkat dropout tertinggi.
   - Insight: Program studi Nursing memiliki jumlah lulusan terbanyak secara signifikan, menunjukkan tingkat kelulusan yang tinggi di jurusan ini. Sebaliknya, beberapa jurusan seperti Biofuel Production Technologies dan Informatics Engineering memiliki jumlah mahasiswa yang sangat sedikit, dengan distribusi status yang tidak merata atau condong ke dropout dan enrolled.


## Menjalankan Sistem Machine Learning
Langkah-langkah menggunakan sistem machine learning berbasis Extra Trees Classifier	adalah sebagai berikut.

1. Membuka link: https://academics-insight-jaya-jaya-institute-by-zainal.streamlit.app/

2. Tampilan Awal
<center><img src="image\home.jpg" alt="alt text" width="whatever" height="whatever"></center>

3. Memilih "Predict Massal" pada taskbar di sisi kiri.

<center><img src="image\menu.jpg" alt="alt text" width="whatever" height="whatever"></center>

3. Tekan tombol **Browse File** untuk mengupload file dataset.

<center><img src="image\prediksi.jpg" alt="alt text" width="whatever" height="whatever"></center>

4. Hasil prediksi akan tampil di bagian bawah.
<center><img src="image\hasil1.jpg" alt="alt text" width="whatever" height="whatever"></center>

<center><img src="image\hasil2.jpg" alt="alt text" width="whatever" height="whatever"></center>

## Conclusion
Berdasarkan analisis dan pemodelan yang dilakukan, ditemukan beberapa insight penting:
- Mahasiswa yang **tidak membayar tepat waktu**, **tidak menerima beasiswa**, dan **memiliki tunggakan** cenderung memiliki kemungkinan lebih tinggi untuk dropout.
- Jurusan tertentu seperti *Veterinary Nursing* dan *Social Service* memiliki dropout rate yang signifikan.
- Mahasiswa laki-laki menunjukkan angka dropout yang lebih tinggi dari perempuan, yang bisa menjadi perhatian lebih lanjut dari pihak institusi.


### Rekomendasi Action Items
Berikut beberapa rekomendasi yang dapat diambil oleh Jaya Jaya Institut:
- 🎯 **Targeted Counseling:** Fokuskan pendampingan akademik pada mahasiswa dengan kombinasi faktor risiko (utang, tanpa beasiswa, tidak tepat waktu membayar).
- 💸 **Pemberian Beasiswa Adaptif:** Perluas cakupan beasiswa untuk mahasiswa dari jurusan atau latar belakang rentan.
- 📊 **Monitoring Berkala via Dashboard:** Gunakan dashboard secara rutin untuk melihat tren dropout dan intervensi dini.
- 🧑‍🏫 **Evaluasi Jurusan Berisiko:** Audit internal terhadap jurusan dengan tingkat dropout tinggi untuk mengevaluasi kurikulum, beban studi, atau dukungan dosen.
- 🛠️ **Peningkatan Sistem Informasi Akademik:** Integrasi sistem prediksi dropout ke dalam sistem akademik yang sudah ada agar dapat langsung memberikan alert.
