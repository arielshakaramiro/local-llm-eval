# Evaluasi LLM Lokal: Qwen2.5 (GGUF) pada Tugas Berbahasa Indonesia

*Also available in [English](README.md).*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arielshakaramiro/local-llm-eval/blob/main/notebooks/local_llm_eval.ipynb)
[![Tests](https://github.com/arielshakaramiro/local-llm-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/arielshakaramiro/local-llm-eval/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Evaluasi kecil dan reproducible untuk model open-weight yang di-hosting sendiri (self-hosted) lewat `llama-cpp-python`, tanpa API eksternal. Eksperimen dijalankan di runtime GPU Google Colab. Repo ini menjawab dua pertanyaan praktis:

1. **Seberapa besar pengaruh ukuran model?** Qwen2.5-1.5B vs 3B (keduanya GGUF Q8) diuji pada 30 pertanyaan berbahasa Indonesia, dengan akurasi, latensi, dan throughput diukur pada kondisi yang sama.
2. **Bisakah model kecil menghasilkan output terstruktur yang bisa diandalkan?** Teks OCR dari gambar teknik dan BOM diubah jadi JSON, membandingkan decoding biasa (prompt saja) dengan **schema-constrained decoding**.

## Sekilas

![Akurasi dengan interval 95% dan latensi rata-rata per model](results/benchmark_accuracy.png)

**Temuan:** Qwen2.5-3B mendapat skor 80,0% (95% CI 62,7–90,5%) dibanding 1,5B yang 66,7% (95% CI 48,8–80,8%) di GPU NVIDIA A100. Interval kepercayaan keduanya tumpang tindih, dan uji eksak McNemar pada pertanyaan berpasangan menghasilkan p=0,29. Jadi sampel 30 pertanyaan ini belum membuktikan adanya perbedaan nyata. Kedua model punya latensi yang hampir sama (0,2 detik/jawaban).

## Yang membuat proyek ini berbeda

- **Statistik yang sesuai untuk sampel kecil.** Interval Wilson 95% untuk akurasi dan uji eksak McNemar untuk perbandingan berpasangan, supaya perbedaan antar model tidak di-overclaim.
- **Penilaian yang tidak menghukum jawaban benar.** Grup regex (`must` / `forbid`) menerima jawaban benar yang ditulis sebagai kalimat lengkap, berbeda dari pencocokan persis (exact match).
- **Dataset yang memvalidasi dirinya sendiri.** Setiap soal disertai contoh jawaban benar dan salah, dicek otomatis lewat test suite.
- **Output terstruktur dievaluasi, bukan sekadar dilihat sekilas.** Tingkat validitas JSON, kesesuaian skema secara ketat, dan akurasi per-field pada sampel berlabel.
- **Kode teruji dengan CI.** Logika inti ada di `src/llm_eval.py`; test suite pytest jalan otomatis setiap push, tanpa perlu mengunduh model apa pun.
- **Environment dicatat bersama hasil.** GPU, versi library, dan apakah GPU offload benar-benar aktif disimpan di `results/environment.json`, sehingga setiap angka waktu bisa dilacak ke hardware asalnya.

## Struktur proyek

```
local-llm-eval/
├── notebooks/local_llm_eval.ipynb   # menjalankan kedua eksperimen, menulis hasil ke results/
├── src/llm_eval.py                  # loading model, chat, penilaian, statistik, ekstraksi
├── data/
│   ├── questions_id.json            # 30 soal: 10 mudah / 10 sedang / 10 sulit
│   └── ocr_samples.json             # 8 sampel OCR berlabel manual (4 bersih, 4 anomali)
├── scripts/update_readme.py         # mengisi tabel hasil di README dari folder results/
├── tests/                           # test suite pytest (tidak perlu model)
├── .github/workflows/tests.yml      # CI
├── results/                         # CSV, tabel Markdown, grafik, dan environment dari run terakhir
└── requirements.txt
```

## Metode

**Model.** `Qwen2.5-1.5B-Instruct` dan `Qwen2.5-3B-Instruct`, GGUF 8-bit, masing-masing dimuat sekali dengan context 4096 token. Decoding greedy (`temperature=0`), chat template bawaan model, dan satu panggilan pemanasan sebelum pengukuran waktu dimulai.

**Benchmark.** Aritmatika dan fakta (mudah), penjelasan singkat yang dinilai lewat grup kata kunci (sedang), dan penalaran multi-langkah (sulit). Pertanyaan sengaja dibuat dalam bahasa Indonesia. Metriknya: akurasi dengan interval Wilson 95%, latensi rata-rata, token per detik, dan jumlah jawaban yang terpotong.

**Ekstraksi terstruktur.** Setiap sampel OCR harus menghasilkan `components` (part number, deskripsi, material, kuantitas, pemasok, dimensi) plus daftar `anomalies`. Dua mode dijalankan pada sampel yang sama:

| Mode | Cara JSON diperoleh |
|---|---|
| `prompt_only` | Prompt menjelaskan bentuk JSON yang diharapkan; output di-parse setelahnya |
| `schema_constrained` | Prompt yang sama plus JSON Schema lewat `response_format`, membatasi token apa saja yang boleh dihasilkan |

Dinilai per sampel: JSON valid, kesesuaian skema secara ketat (misalnya `quantity` harus berupa integer), part number benar, kuantitas benar, dan tanda anomali yang tepat (`anomalies` diisi hanya kalau catatan itu memang bermasalah). Output yang gagal di-parse dianggap salah di semua metrik.

## Hasil

Tabel di bawah dihasilkan oleh `scripts/update_readme.py` dari file di `results/`.

### Benchmark (30 pertanyaan)

<!-- BENCHMARK_SUMMARY:START -->
| model           |   n |   correct |   accuracy_pct |   ci95_low_pct |   ci95_high_pct |   mean_latency_s |   mean_tokens_per_s |   truncated |
|:----------------|----:|----------:|---------------:|---------------:|----------------:|-----------------:|--------------------:|------------:|
| Qwen2.5-1.5B Q8 |  30 |        20 |           66.7 |           48.8 |            80.8 |              0.2 |               214   |           0 |
| Qwen2.5-3B Q8   |  30 |        24 |           80   |           62.7 |            90.5 |              0.2 |               152.7 |           0 |
<!-- BENCHMARK_SUMMARY:END -->

### Akurasi per tingkat kesulitan (%)

<!-- BENCHMARK_BY_LEVEL:START -->
| level   |   Qwen2.5-1.5B Q8 |   Qwen2.5-3B Q8 |
|:--------|------------------:|----------------:|
| easy    |                90 |             100 |
| medium  |                30 |              60 |
| hard    |                80 |              80 |
<!-- BENCHMARK_BY_LEVEL:END -->

### Ekstraksi terstruktur (8 sampel, angka dalam %)

<!-- EXTRACTION_SUMMARY:START -->
| model           | mode               |   parsed |   schema_ok |   part_no_ok |   quantity_ok |   anomaly_ok |   mean_latency_s |   n |
|:----------------|:-------------------|---------:|------------:|-------------:|--------------:|-------------:|-----------------:|----:|
| Qwen2.5-1.5B Q8 | prompt_only        |      100 |         100 |         87.5 |          87.5 |         62.5 |              0.4 |   8 |
| Qwen2.5-1.5B Q8 | schema_constrained |      100 |         100 |         87.5 |          87.5 |         87.5 |              3.4 |   8 |
| Qwen2.5-3B Q8   | prompt_only        |      100 |         100 |         87.5 |          87.5 |        100   |              0.8 |   8 |
| Qwen2.5-3B Q8   | schema_constrained |      100 |         100 |         87.5 |          87.5 |         62.5 |              4.9 |   8 |
<!-- EXTRACTION_SUMMARY:END -->

### Environment

<!-- ENVIRONMENT:START -->
- **Accelerator:** NVIDIA A100-SXM4-40GB, 40960 MiB
- **GPU offload active:** yes
- **llama-cpp-python:** 0.3.35
- **Python:** 3.13.15
- **Platform:** Linux-6.6.122+-x86_64-with-glibc2.39
<!-- ENVIRONMENT:END -->

### Temuan utama

Pada sampel 30 pertanyaan ini, akurasi 3B yang lebih tinggi (80,0% vs 66,7%) belum signifikan secara statistik: interval kepercayaannya tumpang tindih cukup besar, dan uji eksak McNemar pada pertanyaan berpasangan menghasilkan p=0,29. Kedua model performanya mirip di soal mudah dan sulit, tapi 3B jelas lebih unggul di soal sedang (penjelasan singkat), 60% berbanding 30%. Latensi hampir identik di GPU A100 yang dipakai untuk run ini (~0,2 detik/jawaban untuk keduanya), jadi pada angka-angka ini tambahan akurasi 3B tidak berbiaya latensi tambahan, meski ini kemungkinan berubah di CPU atau GPU dengan memori lebih kecil. Pada ekstraksi, baik prompt_only maupun schema_constrained sudah menghasilkan JSON 100% valid dan sesuai skema pada run ini, jadi efek utama skema di sini bukan memperbaiki JSON yang rusak, melainkan memperlambat runtime sekitar 5–8 kali; efeknya terhadap akurasi deteksi anomali berlawanan arah antar model (1,5B naik 62,5% → 87,5%, 3B turun 100% → 62,5%), pola yang lebih terbaca sebagai sensitivitas terhadap prompt ketimbang manfaat yang bisa diandalkan dari constrained decoding. Keempat kombinasi model×mode secara konsisten salah membaca part number ambigu pada sampel s5 (`XYZ-0O7`, huruf O bersebelahan dengan angka 0) sebagai `XYZ-007`, kesalahan mirip kesalahan baca OCR yang berulang, bukan kebetulan.

## Keterbatasan

- 30 pertanyaan dan 8 sampel ekstraksi tergolong kecil; interval kepercayaan di atas menunjukkan seberapa lebar ketidakpastiannya.
- Penilaian berbasis regex hanya mendekati kebenaran dan bisa melewatkan jawaban valid yang ditulis dengan frasa berbeda.
- Hanya satu kali run greedy per model; waktu eksekusi sangat bergantung pada hardware.
- Hanya kuantisasi Q8 dan satu keluarga model yang diuji, dan bahasa Indonesia adalah satu-satunya bahasa yang dibenchmark.
- Label ekstraksi ditulis manual dan anomalinya sengaja dibuat, bukan diambil dari data produksi nyata.

## Menjalankan ulang

**Colab (disarankan).** Klik badge di atas, pilih runtime GPU, lalu jalankan semua sel. Notebook akan meng-clone repo ini kalau perlu, dan membangun `llama-cpp-python` dengan CUDA (5–10 menit). `pip install` biasa menghasilkan build CPU-only meski di runtime GPU, jadi sel instalasi menangani ini secara eksplisit, dan sel environment akan memperingatkan kalau GPU offload ternyata tidak aktif.

**Di mesin sendiri.**

```bash
git clone https://github.com/arielshakaramiro/local-llm-eval.git
cd local-llm-eval
pip install -r requirements.txt
jupyter notebook notebooks/local_llm_eval.ipynb
```

Untuk GPU NVIDIA, install dengan `CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir -r requirements.txt`. Di Apple Silicon, pakai `CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir -r requirements.txt`. Offload layer akan otomatis aktif setelahnya.

**Setelah run selesai**, perbarui tabel di kedua README:

```bash
python scripts/update_readme.py
```

**Test** (tidak perlu mengunduh model):

```bash
pip install pytest
pytest -q
```

## Model dan lisensi

Bobot model tidak disertakan di repo ini; model diunduh dari Hugging Face saat dijalankan. Setiap model punya lisensinya sendiri. Periksa model card sebelum dipakai untuk keperluan komersial, karena model 1.5B dan 3B mungkin punya ketentuan yang berbeda.

## Lisensi

MIT, lihat [LICENSE](LICENSE).
