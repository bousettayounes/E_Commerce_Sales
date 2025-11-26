# E-commerce Product Data Processor & NLP Categorizer

## 📋 Project Overview
This project is an automated data processing pipeline designed to clean, validate, and enrich raw e-commerce product catalogs.

Dealing with large datasets often involves handling human labeling errors and unstructured text. This tool solves those problems by using Natural Language Processing (NLP) to audit and correct product categories, while simultaneously using Regular Expressions (Regex) to extract structured attributes (Dimensions and Colors) buried within text descriptions. It transforms messy Excel dumps (.xlsb/.xlsx) into clean, analyzed CSV datasets ready for reporting or ingestion.

## Project Description
The codebase is built around two primary data cleaning pillars:

1. NLP-Driven Category Correction
Manual product categorization is prone to error. This project implements a Machine Learning approach to audit the Nature (Category) column:

Profile Generation: It aggregates all descriptions within existing categories to build a "vocabulary profile" for each product type.

Vectorization & Similarity: Using TfidfVectorizer (Term Frequency-Inverse Document Frequency), it converts text into numerical vectors. It then calculates the Cosine Similarity between a specific product and every category profile.

Auto-Correction: If a product is statistically more similar to a different category than the one it is currently assigned to (based on a calculated confidence threshold), the script automatically reassigns it to the correct category.

2. Feature Extraction (Attribute Mining)
Product dimensions and colors are often trapped in unstructured description fields (e.g., "Table measuring 120x60cm in dark anthracite..."). This tool extracts them into dedicated columns:

Dimension Extraction: Utilizes complex Regex patterns to identify various formats, including standard dimensions (120x60x40), labeled units (H150, Diam: 20), and standalone metric units.

Color Normalization: Uses a dictionary-based mapping system to detect specific color variations (e.g., "Mustard", "Lemon", "Ochre") and standardize them into parent color families (e.g., "Yellow"). It is optimized to handle French accents and nuances.

3. High-Performance File Handling
Optimized to load Binary Excel files (.xlsb) for faster processing of large datasets.

Includes robust logging to track the cleaning process, progress percentages, and error handling in real-time.

## ✨ Key Features

### 1. Robust Data Loading
- Supports both standard Excel (`.xlsx`) and Binary Excel (`.xlsb`) formats.
- Implements error handling and logging for file operations.

### 2. Smart Category Verification (NLP)
- **Profile Construction:** Aggregates descriptions of verified products to create a textual profile for each category.
- **TF-IDF Vectorization:** Converts text descriptions into numerical vectors using `TfidfVectorizer` (1-3 ngrams).
- **Cosine Similarity:** Calculates the similarity between a specific product and category profiles to detect miscategorized items.
- **Auto-Correction:** Suggests and applies the correct category if a high confidence threshold is met.

### 3. Attribute Extraction
- **Dimensions:** Extracts complex dimension strings (e.g., `12x45x56`, `H150`, `30cm`) using advanced Regex patterns.
- **Colors:** Extracts colors based on a comprehensive dictionary mapping.
  - handles accents (`é`, `è`).
  - Groups variations (e.g., "Mustard", "Lemon" -> `Jaune`).
  - Context-aware extraction (looks for "Couleur:" context if direct keywords are missing).

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/bousettayounes/E_Commerce_Sales)