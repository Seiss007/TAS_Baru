# ML Studio

Aplikasi Streamlit bertema web hiburan (gelap, aksen merah dan emas, judul Bebas Neue, teks Inter) untuk eksperimen machine learning.

## Fitur

| Area | Isi |
|---|---|
| Mode | Regresi, Klasifikasi, Clustering, Deep Learning |
| Dataset scikit-learn | Diabetes, California Housing, Iris, Breast Cancer, Wine |
| Dataset UCI | Abalone, Auto MPG, Wine Quality, Heart Disease, Seeds, Banknote Authentication |
| Dataset real-time | Cuaca per jam (Open-Meteo, 8 kota) dan gempa bumi 30 hari (USGS, regresi dan klasifikasi). Cache 5 menit, tombol perbarui di sidebar |
| Clustering | K-Means (dengan grafik elbow dan silhouette), Agglomerative, DBSCAN. Peta PCA, profil cluster, metrik Silhouette, Davies-Bouldin, Calinski-Harabasz, dan ARI terhadap label asli |
| Deep learning | MLP PyTorch (jumlah layer, neuron, dropout, epoch, learning rate, batch) dengan kurva loss per epoch |
| Komentar publik | Disimpan di MongoDB. Tanpa konfigurasi, otomatis memakai file lokal `comments_local.json` |
| LLM Groq | Analisis sentimen tiap komentar, auto-summary semua komentar, dan penjelasan hasil model |

## Menjalankan

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # lalu isi nilainya
streamlit run app.py
```

Aplikasi tetap berjalan tanpa MongoDB dan Groq (komentar jatuh ke file lokal, fitur AI menampilkan petunjuk).

## Konfigurasi

- **MongoDB**: buat cluster gratis di MongoDB Atlas, tambahkan IP Anda di *Network Access*, lalu isi `MONGODB_URI` dan `MONGODB_DB`. Koleksi `comments` dan indeksnya dibuat otomatis.
- **Groq**: buat API key di https://console.groq.com/keys lalu isi `GROQ_API_KEY`. Model bawaan `llama-3.3-70b-versatile`, dapat diganti lewat `GROQ_MODEL`.
- Untuk deployment di Streamlit Community Cloud, isi nilai yang sama pada menu *Secrets*.

## Struktur

```
app.py               # antarmuka dan alur utama
studio/catalog.py    # dataset sklearn, UCI, real-time
studio/models.py     # supervised, clustering, PyTorch
studio/charts.py     # grafik Plotly bertema gelap
studio/storage.py    # komentar: MongoDB + fallback lokal
studio/llm.py        # Groq: sentimen, ringkasan, insight
studio/styles.py     # CSS dan blok HTML
```

## Catatan

- Data UCI dan real-time membutuhkan internet. Jika gagal dimuat, aplikasi menampilkan pesan error dan dataset lain tetap bisa dipakai.
- Teks komentar di-escape sebelum ditampilkan, dan prompt LLM memperlakukan komentar sebagai data, bukan instruksi.
- Agglomerative memakai sampel acak maksimal 5.000 baris agar hemat memori.
