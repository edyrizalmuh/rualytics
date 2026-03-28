# Pengenalan Macro di Microsoft Excel



## 1. Pengenalan Macro
Bayangkan Anda setiap hari bekerja dengan Excel dan harus melakukan rutinitas yang sama, misalnya merapikan data, mewarnai header tabel, lalu menyimpan file. Ketiga langkah itu Anda lakukan setiap hari dengan urutan yang sama. Pekerjaan itu mungkin hanya butuh 2 menit, tapi jika dilakukan setiap hari selama setahun, totalnya bisa mencapai lebih dari 8 jam terbuang untuk hal yang berulang.

**Macro adalah cara Excel "mengingat" dan "mengulang" serangkaian perintah secara otomatis.** Analoginya sepertinya tombol rekam pada video recorder di HP Anda. Anda hanya perlu menekan tombol rekam, maka Excel akan merekam setiap langkah yang Anda lakukan, kemudian tekan tombol stop ketika serangkaian perintah tersebut sudah selesai. Selanjutnya, Anda tidak perlu lagi melakukan rangkaian perintah tersebut satu per satu, cukup tekan tombol "play", maka Excel akan melakukan rangkaian perintah tersebut.

**Macro berguna untuk pekerjaan yang bersifat berulang, memakan waktu, dan rentan kesalahan manusia**, antara lain:
- Memformat laporan: mengatur font, warna, border, dan ukuran kolom secara otomatis
- Membersihkan data: menghapus baris kosong, menghilangkan spasi berlebih, atau menyeragamkan format teks
- Membuat ringkasan: menyalin data dari beberapa sheet ke satu sheet rekapitulasi
- Mencetak dokumen: mengatur area cetak dan langsung mencetak dengan satu klik

Sebagai ilustrasi, bayangkan Anda harus mencetak setiap sheet dalam workbook Anda dengan beberapa aturan tertentu, seperti ukuran kertas, margin, warna heading, dan seterusnya. Bagaimana Anda akan melakukannya jika seandainya Anda punya 100 sheet? Anda bisa melakukannya satu per satu, tapi tentu akan sangat memakan waktu. Cara yang lebih cerdas adalah dengan merekam prosesnya menggunakan macro, sehingga Anda hanya perlu melakukannya sekali.





## 2. Tahap Persiapan
### 2.1 Mengaktifkan tab Developer di Ribbon
Untuk menggunakan macro, Anda harus memastikan tab Developer sudah aktif. 

{{< image src="/images/figure1.png" caption="Tab Developer di Microsoft Excel" >}}

Secara default, tab Developer tidak ditampilkan di Excel. Jika Anda tidak bisa menemukan tab ini, silakan aktifkan terlebih dahulu dengan langkah-langkah berikut:

1. Klik menu **File → Options**
2. Akan muncul window berikut ini. Pada panel kiri, pilih **Customize Ribbon**. Setelah itu, centang kotak **Developer**.
   {{< image src="/images/figure2.png" caption="File → Options → Customize Ribbon → Centang Developer → OK" >}}
3. Klik OK.

{{< admonition type="note" title="Note" open=true >}}
Langkah ini hanya perlu dilakukan sekali. Tab **Developer** akan tetap muncul setiap kali Excel dibuka. Uncheck kotak **Developer** untuk menonaktifkannya.
{{< /admonition >}}



### 2.2 Format .xlsm (*Excel Macro-Enabled*)

File Excel biasa, yaitu file berformat .xlsx, tidak bisa menyimpan Macro. Oleh karena itu, jika Anda menggunakan Macro dalam sebuah workbook, maka Anda harus menyimpannya sebagai file .xlsm (*Excel Macro-Enabled Workbook*). Anda cukup menyimpannya seperti biasa menggunakan **File → Save/Save As**, kemudian pada bagian **Save as Type**, pilih **Excel Macro-Enabled Workbook (*.xlsm)** kemudian **Save**.

{{< image src="/images/figure3.png" caption="Save as xlsm file" >}}



### 2.3 Mengatur Keamanan Macro
Excel secara default memblokir Macro karena alasan keamanan. Sebab, Macro yang desainnya kurang baik dapat merusak data. Oleh karena itu, jika Anda membuka file .xlsm, Anda bisa jadi melihat warning berikut ini.

{{< image src="/images/figure4.png" caption="Security Warning yang disebabkan oleh Macro" >}}

Anda dapat mengklik tombol **Enable Content** untuk menggunakan Macro. Selain itu, Anda dapat mengubah pengaturan terkait Macro di tab **Developer → Macro Security**.

{{< image src="/images/figure5.png" caption="Macro Security Setting di Tab Developer" >}}

Setelah itu, akan muncul window berikut ini.

{{< image src="/images/figure6.png" caption="Macro Settings → OK" >}}

Terdapat empat pilihan:

1. *Disable VBA macros without notification*: semua macros diblokir total tanpa notifikasi
2. *Disable VBA macros with notification*: semua macros diblokir, tetapi disertai notifikasi seperti gambar sebelumnya
3. *Disable VBA macros except digitally signed macros*: semua macros diblokir kecuali macros yang telah diverifikasi menggunakan sertifikat digital
4. *Enable VBA macros (not recommended; potentially dangerous code can run)*: semua macros langsung berjalan tanpa konfirmasi, hanya untuk file pribadi

{{< admonition type="warning" title="Warning" open=true >}}
Berhati-hatilah dalam menggunakan Macro dari file yang Anda peroleh dari sumber yang tidak dikenal atau tidak dipercaya.
{{< /admonition >}}





## 3. Merekam Macro
### 3.1 Record Macro
Record macro adalah fitur yang memungkinkan Excel untuk merekam setiap langkah yang Anda lakukan, lalu mengubahnya menjadi Macro secara otomatis. Ini merupakan cara termudah untuk membuat macro di Excel. Prinsipnya mudah, Anda cukup mengklik tombol rekam, lakukan setiap langkah yang Anda perlukan, kemudian klik tombol stop. Excel kemudian akan menyimpan langkah-langkah tadi sebagai code dalam bahasa pemrograman Visual Basic. Berikut adalah langkah-langkah yang dapat Anda lakukan untuk merekam sebuah macro di Excel.

1. Klik tombol **Developer** → klik tombol **Record Macro**
   {{< image src="/images/figure7.png" caption="Record Macro: Langkah 1" >}}
2. Akan muncul kotak dialog berikut.
   {{< image src="/images/figure8.png" caption="Record Macro: Langkah 2" >}}
   
   Terdapat beberapa poin penting dari kotak dialog ini:
   - **Macro Name** merupakan nama macro yang akan dibuat. Nama macro harus dimulai dengan huruf atau *underscore* (_), tidak mengandung spasi atau karakter khusus (!, @, &, $), serta tidak melebihi 255 karakter. Nama macro bersifat unik sehingga tidak boleh ada dua macro dengan nama yang sama. 
{{< admonition type="tip" title="Tips" open=false >}}
Nama seperti macro1 dapat digunakan, tapi sebaiknya gunakan nama yang lebih deskriptif yang menjelaskan apa yang dilakukan oleh macro tersebut.
{{< /admonition >}}
   - **Shortcut key** memungkinkan Anda untuk memberikan shortcut tersendiri untuk setiap macro yang Anda buat. Kolom ini bersifat opsional sehingga Anda dapat mengosongkannya saat pertama kali membuat macro. Selanjutnya, Anda dapat menambahkan shortcut dengan mengedit macro yang sudah ada.
{{< admonition type="tip" title="Tips" open=false >}}
Jangan gunakan shortcut yang sudah umum seperti `Ctrl+c`, `Ctrl+v`, dan seterusnya.
{{< /admonition >}}
   - **Store macro in** mengatur lokasi penyimpanan macro, yaitu di mana saja macro tersebut berlaku. Kolom ini memiliki tiga pilihan:
     - **This workbook**: macro akan disimpan dan hanya akan berlaku di dalam workbook yang sedang aktif.
     - **New workbook**: macro akan disimpan dan hanya akan berlaku di dalam workbook baru. Pilihan ini akan membuka file .xlsm baru.
     - **Personal macro workbook**: macro akan disimpan sebagai macro global sehingga Anda dapat menggunakannya di semua workbook.
   - **Description** merupakan kolom dimana Anda dapat memberikan penjelasan atau catatan singkat terkait fungsi macro yang Anda buat.
3. Setelah Anda mengisi semua kolom yang diperlukan, klik OK, maka Excel akan mulai merekam semua langkah yang Anda lakukan. Anda dapat memastikan apakah Excel sudah mulai merekam atau belum dengan melihat tombol "Record macro" sebelumnya. Apabila "Record macro" sudah berubah menjadi "Stop Recording", maka Excel sedang merekam langkah-langkah yang akan Anda lakukan.
    {{< image src="/images/figure9.png" caption="Record Macro: Langkah 3" >}}

    Anda juga bisa mengecek dengan memastikan ada tidaknya tombol persegi di pojok kiri bawah. Jika ada, berarti Excel sedang merekam.
    {{< image src="/images/figure10.png" caption="Record Macro: Langkah 3" >}}

4. Lakukan semua langkah yang Anda perlukan, misalnya mewarnai header, mengatur font, dan sebagainya.
5. Setelah selesai, klik tombol "Stop recording" di tab **Developer** atau tombol persegi di pojok kiri bawah.

{{< admonition type="tip" title="Tips" open=true >}}
Sebelum mulai merekam, rencanakan terlebih dahulu langkah-langkah yang akan dilakukan. Semua gerakan — termasuk klik yang tidak perlu — akan ikut terekam.
{{< /admonition >}}



### 3.2 Absolute vs Relative Reference
Excel secara default akan merekam macro menggunakan *absolute reference*. Artinya, saat merekam macro, Excel akan mengingat posisi cell yang sama persis saat direkam. Sebagai ilustrasi, misalnya Anda ingin memberikan *fill color* pada baris heading di cell A1 hingga A10. Jika saat merekam macro cell aktif berada di A10, maka ketika Anda memanggil macro yang sama, cell yang akan terisi tetap A10, meskipun Anda sudah memindahkan cell aktif ke A1. Fitur ini cocok digunakan jika data selalu berada di posisi yang tetap, misalnya ketika Anda ingin menghitung total dari sebuah tabel di mana cell untuk nilai total sudah ditentukan, misalnya di cell Z46, maka penggunaan *absolute reference* sangat direkomendasikan.

Meskipun begitu, seringkali terdapat beberapa kasus di mana *absolute reference* kurang cocok untuk digunakan. Misalnya, Anda ingin mengubah format 4 cell ke kanan dari cell yang sedang aktif, maka penggunaan *absolute reference* tidak dapat memenuhi keperluan Anda ini. Dalam kasus ini, Anda perlu menggunakan *relative reference*. *Relative reference* memungkinkan macro untuk bekerja relatif terhadap posisi kursor saat dijalankan, cocok digunakan jika posisi data bisa berubah-ubah.

Untuk mengaktifkan *relative reference*, Anda cukup mengklik tombol **Use Relative Reference** di tab **Developer** sebelum Anda mulai merekam.

{{< image src="/images/figure11.png" caption="Relative Reference" >}}


## 4 Mengelola Macro
Setelah membuat beberapa macro, Anda perlu tahu bagaimana cara mengaksesnya kembali. Excel menyediakan satu tempat terpusat untuk melihat semua macro yang tersimpan. Anda cukup klik tab **Developer → Macros**.
{{< image src="/images/figure12.png" caption="Mengakses daftar macro yang tersimpan" >}}

Selanjutnya akan muncul window berikut.
   {{< image src="/images/figure13.png" caption="Daftar macro yang tersimpan" >}}

Berikut adalah penjelasan terkait tombol-tombol dalam window tersebut.
- **Macros in** merupakan dropdown dengan dua nilai: *This workbook* dan *All Open Workbooks*, merupakan menu untuk memfilter daftar macros yang ditampilkan, apakah Excel menampilkan hanya macros dari workbook Anda saat ini atau apakah Excel harus menampilkan semua macros yang tersimpan di semua Workbooks yang sedang terbuka.
- **Run** merupakan perintah untuk menjalankan sebuah macro
- **Step into** merupakan alat debugging di mana Anda dapat menjalankan sebuah macro satu baris per waktu (per tahap, tidak langsung dieksekusi keseluruhannya)
- **Edit** memungkinkan Anda mengedit macro yang sudah tersimpan. Anda perlu menggunakan bahasa pemrograman Visual Basic.
- **Delete** digunakan untuk menghapus macro
- **Options** memungkinkan Anda untuk mengedit shortcut dan deskripsi macro yang sudah tersimpan.
   {{< image src="/images/figure14.png" caption="Menu Options pada Daftar Macro" >}}


## 5 Menjalankan Macro Hasil Rekaman
Ada tiga cara untuk menjalankan macro yang sudah direkam, yaitu dengan menggunakan shortcut, menggunakan menu macro di tab Developer, dan menggunakan tombol di worksheet.

### 5.1 Shortcut Keyboard
Cara ini merupakan cara yang paling cepat dan sederhana. Cukup tekan kombinasi shortcut yang sudah Anda atur sebelumnya, maka macro akan langsung dieksekusi. Cara ini sangat cocok untuk macro yang sering digunakan, sehingga Anda tidak perlu membuka menu setiap kali ingin menjalankannya.


### 5.2 Menu Macro
Sebelumnya Anda sudah melihat daftar macro yang tersimpan di sebuah workbook. Anda cukup memilih macro yang ingin dijalankan, kemudian klik **Run**.

{{< image src="/images/figure15.png" caption="Menjalankan macro melalui menu macro" >}}

{{< admonition type="note" title="Note" open=false >}}
Ini merupakan salah satu alasan mengapa penamaan macros disarankan menggunakan nama yang deskriptif, sehingga pengguna dapat langsung membedakan macro yang perlu digunakan hanya dari nama saja.
{{< /admonition >}}



### 5.3 Menggunakan Tombol
Tombol merupakan cara paling intuitif untuk menjalankan Macro, terutama jika file macro yang Anda buat akan digunakan oleh orang lain. Idenya sederhana, Anda cukup membuat tombol di worksheet kemudian dihubungkan dengan macro yang Anda buat. Setiap kali tombol ditekan, macro yang terhubung tadi akan tereksekusi.

Berikut adalah langkah-langkah untuk membuat tombol macro:
1. Klik tab **Developer → Insert → Button (Form Control)**
   {{< image src="/images/figure16.png" caption="Membuat button (form control)" >}}

   {{< admonition type="note" title="Note" open=false >}}
  Ada dua jenis button yang bisa Anda sisipkan: Form Controls dan ActiveX controls. Form control merupakan kontrol bawaan Excel, lebih sederhana, kompatibel dengan Windows dan Mac, serta langsung bisa digunakan dengan record macro tanpa perlu dituliskan menggunakan VBA (*Visual Basic for Applications*). Sementara itu, ActiveX controls hanya kompatibel dengan Windows dan lebih kompleks sebab perlu dituliskan menggunakan VBA editor.
    {{< /admonition >}}

2. Kursor Anda akan berubah menjadi tanda **+**
3. Klik dan seret di area worksheet untuk menggambar tombol sesuai ukuran yang Anda inginkan, atau cukup klik sekali di area yang Anda inginkan.
4. Kotak dialog **Assign Macro** akan muncul secara otomatis.
   {{< image src="/images/figure17.png" caption="Menghubungkan macro ke tombol" >}}
5. Klik **Record** dan mulai merekam. Atau pilih macro dari daftar yang sudah ada.
6. Jika ingin mengganti macro, klik kanan pada tombol, kemudian pilih **Assign macro**.
   {{< image src="/images/figure18.png" caption="Mengubah macro yang terhubung" >}}
7. Setelah tombol dan macro terhubung, pengguna cukup menekan tombol saja untuk menjalankan macro.






## 6. Pengenalan *Visual Basic for Applications* (VBA)
VBA adalah bahasa pemrograman bawaan Microsoft yang terintegrasi dengan aplikasi Office, bukan hanya Microsoft Excel, tapi juga Microsoft Word, Access, dan Powerpoint. Bahasa pemrograman ini digunakan untuk menjalankan macro. Singkatnya, setiap kali Anda merekam sebuah macro, Excel akan mengonversi setiap langkah yang Anda rekam menjadi kode VBA.

Anda dapat memeriksa dan mengedit kode VBA yang Anda rekam melalui VBA Editor. VBA editor dapat diakses dengan menekan tombol **Alt + F11** atau dengan mengklik tab **Developer → Visual Basic**.
{{< image src="/images/figure19.png" caption="Mengakses Visual Basic Editor" >}}

Berikut adalah tampilan dari VBA Editor.
{{< image src="/images/figure20.png" caption="Jendela Visual Basic Editor" >}}

Terdapat beberapa bagian utama yang perlu Anda ketahui dari jendela VBA Editor:
- **Project Explorer (Ctrl + R):** menampilkan struktur file dan semua module yang ada
- **Properties Window (F4):** menampilkan properti dari objek yang dipilih
- **Code Window :** area utama tempat kode VBA ditulis dan dibaca
- **Immediate Window (Ctrl + G):** area untuk menguji kode

{{< admonition type="tip" title="Tips" open=false >}}
Jika salah satu panel tidak terlihat, Anda bisa mengaktifkan melalui menu **View** di VBA Editor atau menggunakan shortcut untuk masing-masing panel.
{{< /admonition >}}



