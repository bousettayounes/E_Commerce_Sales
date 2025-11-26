# E-commerce_USE_CASE

## 📋 Project Overview
This project provides an **automated data processing pipeline** designed to **clean, validate, and enrich raw e-commerce product catalogs**.  

Large product datasets often contain **human labeling errors** and **unstructured text**, making analysis challenging. This solution addresses these issues using **Natural Language Processing (NLP)** for category auditing and correction, and **Regular Expressions (Regex)** for structured attribute extraction (Dimensions and Colors). It transforms messy Excel dumps (`.xlsb` / `.xlsx`) into **clean, analyzed CSV datasets** ready for reporting and to extract insights.

---

## 🔍 Core Functionality

### 1. NLP-Driven Category Correction
Manual product categorization is error-prone. This tool implements a **machine learning approach** to audit the *Nature* (Category) column:

- **Profile Generation:** Aggregates all product descriptions within each category to build a "vocabulary profile".
- **Vectorization & Similarity:** Uses `TfidfVectorizer` to convert text into numerical vectors, then calculates **Cosine Similarity** between products and category profiles.
- **Auto-Correction:** If a product aligns more closely with a different category than its current one (based on a confidence threshold), the script automatically reassigns it.

### 2. Feature Extraction (Attribute Mining)
Dimensions and colors are often buried in unstructured text. This tool extracts them into **dedicated columns**:

- **Dimension Extraction:** Advanced Regex patterns detect standard dimensions (`120x60x40`), labeled units (`H150`, `Diam: 20`), and standalone metric values.
- **Color Normalization:** Uses a **dictionary-based system** to standardize color variations (e.g., "Mustard", "Lemon" → `Yellow`). Handles **French accents** and context-aware color mentions.

### 3. High-Performance File Handling
- Loads **Binary Excel files (`.xlsb`)** efficiently for large datasets.
- Implements **robust logging** to track progress, detect errors, and monitor the cleaning process in real-time.

---

## 🛠️ Pipeline Steps

1. **Load Data** – Import "Ecommerce_sales.xlsb" file.  
2. **Clean Text** – Remove accents, extra spaces, and unwanted characters.  
3. **Generate Category Profiles** – Aggregate descriptions by existing categories.  
4. **Vectorize Text** – Transform descriptions into TF-IDF vectors.  
5. **Compute Similarity** – Compare products to category profiles using Cosine Similarity.  
6. **Correct Categories** – Reassign products if similarity suggests a different category.  
7. **Extract Dimensions** – Parse product descriptions for measurements.  
8. **Extract Colors** – Normalize color mentions into standard categories.  
9. **Export CSV** – Save cleaned and enriched dataset for analysis or reporting.

---
## ✨ Key Features

### Robust Data Loading
- Supports **`.xlsx`** and **`.xlsb`** formats.
- Includes error handling and progress logging.

### Smart Category Verification (NLP)
- **Profile Construction:** Creates textual profiles from verified product descriptions.
- **TF-IDF Vectorization:** Converts text to numerical vectors (1-3 ngrams).
- **Cosine Similarity:** Detects miscategorized products.
- **Auto-Correction:** Suggests or applies corrections based on confidence thresholds.

### Attribute Extraction
- **Dimensions:** Handles complex formats such as `12x45x56`, `H150`, `30cm`.
- **Colors:** Detects and normalizes color mentions.
  - Supports French accents (`é`, `è`).
  - Groups similar shades (e.g., "Mustard", "Lemon" → `Jaune`).
  - Context-aware extraction (uses "Couleur:" cues if direct keywords are absent).

---
## 📁 Project Structure

```
Ecommerce_Project/
├── Ecommerce_Data/
│   ├── Ecommerce_sales.xlsx
│   └── Exported_Ecommerce_sales.csv
│
├── main.py
├── Notebook_version.ipynb
├── Presentation_des_resultats.mp4
└── README.md
```

---
## 🛠️ Installation
1. Clone the repository:

```bash
git clone https://github.com/bousettayounes/E_Commerce_Sales.git
```
