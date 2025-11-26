# E-commerce Product Data Processor & NLP Categorizer

## 📋 Project Overview
This project provides an **automated data processing pipeline** designed to **clean, validate, and enrich raw e-commerce product catalogs**.  

Large product datasets often contain **human labeling errors** and **unstructured text**, making analysis challenging. This tool addresses these issues using **Natural Language Processing (NLP)** for category auditing and correction, and **Regular Expressions (Regex)** for structured attribute extraction (Dimensions and Colors). It transforms messy Excel dumps (`.xlsb` / `.xlsx`) into **clean, analyzed CSV datasets** ready for reporting or downstream applications.

---

## 🔍 Core Functionality

The codebase revolves around **two main pillars**:

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

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/bousettayounes/E_Commerce_Sales.git
