+++
date = '2026-03-24T06:00:00+08:00'
draft = false
title = 'Pivot Table'
translationKey = "pivot-table"
languages = 'id'
tags = ['Microsoft Excel', 'Pemula']
featuredImage = "images/thumbnail-pivot-table.png"
featuredImagePreview = "images/thumbnail-pivot-table.png"
# series = ['Pengantar Microsoft Excel']
categories = ['Tutorial']
description = "Pelajari cara menggunakan Pivot Table di Excel untuk merangkum dan menganalisis data secara efisien. Bandingkan dengan penggunaan formula seperti COUNTIF dan SUMIF."
summary = "Pivot Table adalah tools Excel yang powerful untuk merangkum, membandingkan, dan menemukan pola dalam data besar. Artikel ini ditujukan bagi pemula, menjelaskan kapan menggunakan Pivot Table vs formula, beserta panduan praktisnya."
+++



## 1. Pendahuluan
Pivot Table adalah fitur di Microsoft Excel yang berguna untuk merangkum, membandingkan, dan menemukan pola dalam dataset besar. Untuk memahami kegunaan Pivot Table, perhatikan [data sampel berikut](data/Employee%20Sample.xlsx).

| No. ID | Full Name        | Job Title              | Department  | Business Unit       | Gender | Ethnicity | Age  | Hire Date  | Annual Salary | Bonus % | City          | Exit Date |
| :----- | :--------------- | :--------------------- | :---------- | :------------------ | :----- | :-------- | :--- | :--------- | :------------ | :------ | :------------ | :-------- |
| 1      | Riley Washington | Director               | Sales       | Speciality Products | Female | Caucasian | 39   | 29/04/2007 | $171,487      | 23%     | United States | Phoenix   |
| 2      | Elena Vang       | Analyst                | Finance     | Corporate           | Female | Asian     | 52   | 19/02/2019 | $55,859       | 0%      | China         | Beijing   |
| 3      | Raelyn Ma        | Sr. Analyst            | Finance     | Speciality Products | Female | Asian     | 33   | 08/10/2015 | $94,876       | 0%      | United States | Miami     |
| 4      | Elena Her        | Account Representative | Sales       | Manufacturing       | Female | Asian     | 62   | 17/09/2006 | $64,669       | 0%      | China         | Chongqing |
| 5      | Gabriel Joseph   | Director               | Engineering | Manufacturing       | Male   | Caucasian | 52   | 28/10/2006 | $187,992      | 28%     | United States | Miami     |

Data sampel ini dibuat oleh [Chris Newman](https://www.thespreadsheetguru.com/author/christhespreadsheetguru-com/) dan menampilkan data karyawan sebuah perusahaan. Bayangkan Anda adalah seorang analis di perusahaan tersebut, dan atasan Anda meminta Anda menganalisis data dengan menjawab pertanyaan-pertanyaan berikut:
- Berapa jumlah karyawan per departemen?
- Berapa jumlah karyawan **aktif** per departemen?
- Berapa total gaji per departemen?
- Berapa rata-rata gaji berdasarkan gender?

Tugas ini memerlukan Anda merangkum data dari tabel besar. Sebagai analis, Anda bisa menyelesaikannya dengan dua pendekatan. Pertama, menggunakan fungsi-fungsi seperti `COUNTIF()`, `SUMIF()`, `AVERAGEIF()`, atau `COUNTIFS()`. Kedua, menggunakan Pivot Table.

Apakah Pivot Table selalu lebih efektif dibanding formula? Tidak selalu. Semua bergantung pada kebutuhan Anda. Pivot Table akan lebih efektif pada kasus-kasus berikut:
- Data berukuran besar (ratusan atau ribuan baris)
- Anda perlu melakukan eksplorasi cepat tanpa perlu menulis banyak formula
- Data sering diperbarui dalam periode tertentu — Anda hanya perlu me-*refresh* Pivot Table tanpa mengubah formula

Selanjutnya akan diberikan perbandingan antara penggunaan formula dan pivot table.




## 2. Penggunaan Formula

### 2.1 Jumlah Karyawan per Departemen

Menghitung jumlah karyawan sama dengan menghitung jumlah baris per departemen. Karena kolom departemen berisi data kategori, Anda bisa menggunakan fungsi `COUNTIF()`. Untuk menghitung karyawan berdasarkan departemen tertentu, gunakan formula:

```excel
=COUNTIF(D:D;"Sales")
```

{{< image src="/images/formula-jumlah-karyawan-per-departemen.png" caption="Jumlah karyawan pada departemen **Sales**" >}}

Dengan logika yang sama, Anda bisa menghitung untuk departemen lain, seperti Finance:

{{< image src="images/formula-jumlah-karyawan-per-departemen2.png" caption="Jumlah karyawan pada departemen **Finance**" >}}

Namun, Anda akan menyadari bahwa harus mengganti nilai departemen secara manual di setiap formula. Cara yang lebih efisien adalah menggunakan **referensi sel**. Alih-alih mengetik "Finance" secara manual, gunakan referensi sel seperti P3 yang berisi nilai "Finance". Lihat perbedaannya di gambar berikut:

{{< image src="images/formula-jumlah-karyawan-per-departemen3.png" caption="Jumlah karyawan pada departemen **Finance** menggunakan *cell referencing*" >}}

Kedua formula menghasilkan hasil yang sama. Referensi sel menjadi lebih berguna saat dikombinasikan dengan fungsi `UNIQUE()`, yang mengekstrak nilai unik dari sebuah kolom. Menggunakan `UNIQUE()` membantu menghindari kesalahan manual seperti:
- Lupa menyebutkan beberapa departemen
- Menyebutkan nama departemen yang sama berkali-kali
- Menyebutkan nama departemen yang tidak ada di tabel

Dengan `UNIQUE()`, Anda hanya perlu membuat formula `COUNTIF()` sekali saja. Nilai kondisinya diambil dari fungsi `UNIQUE()` melalui referensi sel.

Video berikut mengilustrasikan kombinasi penggunaan formula `COUNTIF()`, `UNIQUE()`, dan *cell referencing*.

<figure>
  {{< youtube H2oxyxbRpRk >}}
  <figcaption class="image-caption">Ilustrasi penggunaan <code>COUNTIF()</code>, <code>UNIQUE()</code>, dan <em>cell referencing</em></figcaption>
</figure>

{{< admonition type="note" title="Note" open=true >}}
Nilai 0 pada kolom `UNIQUE()` dapat menunjukkan dua hal: kolom yang dirujuk memiliki cell dengan nilai 0 atau cell kosong. Dalam konteks ini, kolom yang dirujuk (kolom D) mengandung cell kosong setelah baris ke-1000.
{{< /admonition >}}




### 2.2 Jumlah Karyawan Aktif per Departemen

Sering kali, agregasi memerlukan lebih dari satu kondisi. Dalam kasus ini, Anda memerlukan dua kondisi:
1. Hitung per departemen (kolom D)
2. Hitung hanya karyawan aktif — yang tidak memiliki nilai di kolom Exit Date (kolom N)

Untuk kasus ini, gunakan `COUNTIFS()` (perhatikan huruf S di akhir). Fungsi ini mirip dengan `COUNTIF()` tetapi memungkinkan lebih dari satu kondisi:

```excel
=COUNTIFS(D:D;S2;N:N;"")
```

Gambar berikut mengilustrasikan penggunaan `COUNTIFS()` untuk menyelesaikan permasalahan kedua.

{{< image src="images/formula-jumlah-karyawan-aktif-per-departemen.png" caption="Jumlah karyawan **aktif** per departemen" >}}



### 2.3 Total Gaji per Departemen

Data gaji karyawan berada di kolom **Annual Salary (kolom J)**. Berbeda dengan dua kasus sebelumnya yang menggunakan data kategori, kasus ini melibatkan data numerik. Oleh karena itu, gunakan `SUMIF()` alih-alih `COUNTIF()`:

```Excel
=SUMIF(D:D;V2;J:J)
```

{{< admonition type="warning" title="Warning" open=true >}}
Sebelum menggunakan `SUMIF()`, pastikan kolom yang ingin ditotalkan, kolom **Annual Salary (J)**, sudah dalam format *number/currency/accounting*. Jika belum dalam format tersebut, ubah dengan memblok kolom J, kemudian Ctrl+1.
{{< /admonition >}}

<figure>
  {{< youtube fyYUOG_PBxQ >}}
  <figcaption class="image-caption">Ilustrasi penggunaan <code>SUMIF()</code></figcaption>
</figure>



### 2.4 Rata-Rata Gaji per Gender

Untuk mencari rata-rata gaji berdasarkan gender, gunakan fungsi `AVERAGEIF()`. Dalam kasus ini, kondisi didasarkan pada kolom **Gender (kolom F)**:

```Excel
=AVERAGEIF(F:F;Y2;J:J)
```

{{< image src="images/formula-rata-rata-gaji-karyawan-per-gender.png" caption="Rata-rata gaji karyawan berdasarkan gender" >}}


## 3. Pivot Table

Bagian sebelumnya menunjukkan cara menggunakan formula untuk agregasi — tambahkan `IF` atau `IFS` pada fungsi dasar seperti `COUNT`, `SUM`, atau `AVERAGE`. Namun, Anda juga bisa menggunakan Pivot Table untuk mencapai hal yang sama. Ikuti langkah-langkah berikut:

1. **Blok semua data** yang ingin diagregasi (Ctrl+A)
   {{< image src="images/pivot-table-step1.png" caption="Pivot Table: Langkah 1" >}}

2. **Buka menu Insert**, kemudian klik **Pivot Table** di sebelah kiri
   {{< image src="images/pivot-table-step2.png" caption="Pivot Table: Langkah 2" >}}

3. **Dialog box akan muncul**. Perhatikan opsi-opsi berikut:
   {{< image src="images/pivot-table-step3.png" caption="Pivot Table: Langkah 3" >}}
   - **Table/Range**: Menunjukkan data yang akan diagregasi (data yang Anda blok)
   - **New Worksheet/Existing Worksheet**: Pilih tempat hasil Pivot Table disimpan
   - **Add this to the Data Model**: Opsi untuk menambahkan tabel ke database relasional (berguna jika menganalisis beberapa tabel terkait)

4. **Klik OK**, maka akan muncul area Pivot Table kosong dan panel pengaturan Pivot Table di sebelah kanan**
   {{< image src="images/pivot-table-step4.1.png" caption="Pivot Table: Langkah 4 - Area Pivot Table" >}}
   {{< image src="images/pivot-table-step4.2.png" caption="Pivot Table: Langkah 4 - Panel pengaturan" >}}
   
   Panel di sebelah kanan menampilkan semua kolom dari data Anda. Di bawah, terdapat empat area untuk membangun Pivot Table. Seret (*drag*) kolom dari atas ke salah satu area di bawah sesuai kebutuhan.

5. **Untuk menjawab pertanyaan pertama** (jumlah karyawan per departemen), seret kolom **Department** ke area **Rows** dan kolom **Full Name** ke area **Values**
   {{< image src="images/pivot-table-step5.1.png" caption="Pivot Table: Langkah 5" >}}
   
   Excel secara otomatis memilih fungsi yang sesuai. Karena **Full Name** adalah data kategori, Excel menggunakan fungsi `COUNT`. Jika Anda menggunakan **No. ID** (data numerik), Excel akan menggunakan `SUM`.

6. **Ubah fungsi jika diperlukan** dengan mengklik panah kecil di samping nama kolom di area **Values**, kemudian pilih **Value Field Settings**
   {{< image src="images/pivot-table-step6.1.png" caption="Pivot Table: Langkah 6" >}}
   
   Dialog akan muncul. Pilih fungsi yang Anda butuhkan dan sesuaikan format jika diperlukan, lalu klik OK
   {{< image src="images/pivot-table-step6.2.png" caption="Pivot Table: Langkah 6 - Pengaturan fungsi" >}}
   {{< image src="images/pivot-table-step6.3.png" caption="Hasil Pivot Table: Jumlah karyawan per departemen" >}}


## 4. Ringkasan
Anda telah mempelajari dua pendekatan untuk merangkum dan menganalisis data di Excel: **Formula** dan **Pivot Table**.

**Pendekatan Formula** menggunakan fungsi bawaan seperti `COUNTIF()`, `COUNTIFS()`, `SUMIF()`, dan `AVERAGEIF()`. Keuntungan formula:
- Lebih fleksibel untuk perhitungan kompleks
- Hasil dapat ditaruh di mana saja dalam worksheet
- Cocok untuk dataset kecil hingga menengah
- Memberikan kontrol penuh atas rumus yang digunakan

**Pendekatan Pivot Table** secara otomatis mengagregasi data tanpa perlu menulis formula. Keuntungan Pivot Table:
- Lebih cepat untuk eksplorasi data
- Mudah mengubah tata letak tanpa menulis ulang
- Lebih efisien untuk dataset besar
- Otomatis memilih fungsi yang sesuai
- Cukup me-*refresh* saat data diperbarui

**Gunakan Formula jika:**
- Dataset relatif kecil
- Anda memerlukan perhitungan khusus yang kompleks
- Hasil agregasi digunakan dalam perhitungan lebih lanjut

**Gunakan Pivot Table jika:**
- Dataset berukuran besar (ratusan hingga ribuan baris)
- Anda perlu eksplorasi cepat dengan berbagai sudut pandang
- Data sering diperbarui dan Anda ingin hasil otomatis tersinkronisasi
- Anda tidak ingin menulis banyak formula

Pilihan antara keduanya bergantung pada kebutuhan dan preferensi Anda. Sebagai seorang analis, akan sangat berguna jika Anda menguasai kedua teknik ini untuk meningkatkan produktivitas dalam menganalisis data.