+++
date = '2026-03-20T15:57:53+08:00'
draft = true
title = 'Pivot Table 2'
translationKey = "pivot-table"
tags = ['excel']
series = ['Microsoft-Excel']
+++

# Pivot Table
## 📌 Pivot Table Excel (Ringkas)
{{< image src="images.jpeg" caption="Caption here" >}}

- Pivot Table adalah fitur Excel untuk merangkum data besar jadi tabel ringkas.
- Cocok buat analisis: total, rata-rata, hitung, dan kelompok data berdasarkan kolom.

<!--more-->

## 🔧 Cara cepat
1. Pilih range data (termasuk header).
2. Insert → PivotTable.
3. Tarik field:
   - `Rows` untuk kategori (misal: Produk).
   - `Columns` untuk dimensi tambahan (misal: Tahun).
   - `Values` untuk angka (misal: Penjualan).
   - `Filters` untuk saring subset data.

{{< image src="newsCover_2023_7_27_1690442617847-p3oo3l.jpeg" caption="Caption here" >}}

```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()
```

```r
library(tidyverse)

df <- read_csv("data.csv")
head(df)
```

## 💡 Tips
- Gunakan `Value Field Settings` → `Sum/Count/Average`.
- Refresh data dengan klik kanan Pivot → Refresh.
- Pakai `Slicer` untuk interaktif.
