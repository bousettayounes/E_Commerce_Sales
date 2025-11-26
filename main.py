import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re , logging ,unicodedata
from collections import Counter
from pyxlsb import open_workbook


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def load_data(filepath):
    """Load data from an Excel file (.xlsx or .xlsb) with logging & error handling."""

    logging.info(f"Loading file: {filepath}")

    try:
        if filepath.endswith(".xlsb") or filepath.endswith(".xls"):
            logging.info(" XLSB/XLS file is Detected .")

            data = []
            with open_workbook(filepath) as wb:
                with wb.get_sheet(1) as sheet:
                    for row in sheet.rows():
                        data.append([item.v for item in row])

            if not data:
                raise ValueError("The file is empty or could not be read.")

            df = pd.DataFrame(data[1:], columns=data[0])
        
        else:
            logging.info("XLSX file NOT DETECTED , Reading file using pandas.read_excel().")
            df = pd.read_excel(filepath)

        logging.info("File loaded successfully")
        return df

    except Exception as e:
        logging.error(f"FAILED to load File : {e}")
        raise RuntimeError(f"Could not load file '{filepath}'") from e
    

def analyze_categories(df, category_col="Nature"):
    """
    Analyze the distribution of categories and return a DataFrame
    """
    # Check if column exists
    if category_col not in df.columns:
        logging.error(f"Column '{category_col}' does not exist in the DataFrame.")
        return None

    # Count categories
    category_counts = Counter(df[category_col].dropna())
    total_count = sum(category_counts.values())

    # Build the DataFrame
    Category_Distribution = pd.DataFrame({
        'Category': list(category_counts.keys()),
        'Count': list(category_counts.values())
    })

    # Add percentage column
    Category_Distribution['Percentage % '] = round((Category_Distribution['Count'] / total_count) * 100,2)

    # Sort by Count descending
    Category_Distribution = Category_Distribution.sort_values(by='Count', ascending=False).reset_index(drop=True)

    return Category_Distribution

def build_category_profiles(df, nature_col='Nature', description_col="Libellé produit"):
    """
    Construit un profil textuel pour chaque catégorie basé sur les descriptions. Retourne un dictionnaire : {category: texte_concatené_descriptions}
    """
    logging.info("=== Profiles Construction ===")
    
    category_profiles = {}
    
    # Boucle sur toutes les catégories uniques non-null
    for category in df[nature_col].dropna().unique():
        # Filtrer les lignes appartenant à cette catégorie
        category_data = df[df[nature_col] == category]
        
        # Récupérer les descriptions non-null et convertir en string
        descriptions = category_data[description_col].dropna().astype(str)
        
        # Concaténer toutes les descriptions en une seule chaîne
        category_profiles[category] = ' '.join(descriptions)

    logging.info(f"Profils créés pour {len(category_profiles)} catégories")
    return category_profiles

def detect_miscategorized_products(df, nature_col='Nature', description_col='Libellé produit', threshold=0.3):
    """
    Détecte les produits potentiellement mal catégorisés et propose des recatégorisations basées sur la similarité textuelle.
    Paramètres:
    - threshold: seuil de similarité minimum pour considérer qu'un produit est bien catégorisé
    """
    logging.info("=== DÉTECTION DES PRODUITS MAL CATÉGORISÉS ===")
    
    # Construire les profils de catégories
    category_profiles = build_category_profiles(df, nature_col, description_col)
    
    # Préparer les données pour la vectorisation
    categories = list(category_profiles.keys())
    category_texts = [category_profiles[cat] for cat in categories]
    
    # Vectorisation TF-IDF
    logging.info("Vectorisation TF-IDF en cours...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),
        min_df=2,
        stop_words=None  
    )
    
    # Fit sur les profils de catégories
    category_vectors = vectorizer.fit_transform(category_texts)
    
    # Analyser chaque produit
    results = []
    df_copy = df.copy()
    
    logging.info(f"Analyse de {len(df)} produits...")
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            logging.info(f"  Progression : {idx}/{len(df)}")
        
        current_category = row[nature_col]
        description = str(row[description_col]) if pd.notna(row[description_col]) else ""
        
        if pd.isna(current_category) or description == "":
            continue
        
        # Vectoriser la description du produit
        product_vector = vectorizer.transform([description])
        
        # Calculer la similarité avec toutes les catégories
        similarities = cosine_similarity(product_vector, category_vectors)[0]
        
        # Trouver la meilleure catégorie
        best_category_idx = np.argmax(similarities)
        best_category = categories[best_category_idx]
        best_similarity = similarities[best_category_idx]
        
        # Similarité avec la catégorie actuelle
        if current_category in categories:
            current_category_idx = categories.index(current_category)
            current_similarity = similarities[current_category_idx]
        else:
            current_similarity = 0
        
        # Détecter si le produit est mal catégorisé
        is_miscategorized = (best_category != current_category and 
                            best_similarity > current_similarity + 0.1)
        
        if is_miscategorized:
            results.append({
                'index': idx,
                'description': description[:100], 
                'current_category': current_category,
                'current_similarity': round(current_similarity, 3),
                'suggested_category': best_category,
                'suggested_similarity': round(best_similarity, 3),
                'confidence': round(best_similarity - current_similarity, 3)
            })
            
            # Mettre à jour la catégorie dans le DataFrame
            df_copy.at[idx, nature_col] = best_category
    
    logging.info(f"{len(results)} produits mal catégorisés détectés")
    
    return df_copy, pd.DataFrame(results)

def extract_all_dimensions(text: str) -> List[str]:
    """
    Extracts dimensions from the Libellé produit
    """
    if pd.isna(text) or not text:
        return []
    
    text = str(text)
    text_lower = text.lower().replace(',', '.') # Normalize decimals
    
    found_values = []

  # --- 1. MULTI-DIMENSIONS (12x45, 12x45x56) ---
    # On ajoute un groupe optionnel à la fin : (?:\s*[xX/*×]\s*(\d+(?:\.\d+)?))?
    multi_dim_pattern = r'(\d+(?:\.\d+)?)\s*(?:cm|mm|m)?\s*[xX/*×]\s*(\d+(?:\.\d+)?)(?:\s*(?:cm|mm|m)?\s*[xX/*×]\s*(\d+(?:\.\d+)?))?'
    
    for match in re.finditer(multi_dim_pattern, text_lower):
        # On récupère tous les groupes qui ne sont pas None
        parts = [p for p in match.groups() if p]
        
        # Filtre de sécurité (< 3000)
        if len(parts) >= 2 and all(float(p) < 3000 for p in parts):
            found_values.append("x".join(parts))

    # --- 2. LABELED DIMENSIONS (H150, L:200) ---
    labeled_patterns = [
        r'(?:diam[èe]tre|diam\.?|ø)\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r'(?:longueur|length|long\.?|L)\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r'(?:largeur|width|larg\.?|W)\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r'(?:hauteur|height|haut\.?|H)\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r'(?:profondeur|depth|prof\.?|D|P)\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r'(?:epaisseur|épaisseur|thickness|ep\.)\s*[:=]?\s*(\d+(?:\.\d+)?)'
    ]
    for pattern in labeled_patterns:
        for match in re.finditer(pattern, text_lower):
            found_values.append(match.group(1))

    # --- 3. STANDALONE UNITS ---
    # A number followed immediately by a unit
    unit_pattern = r'(\d+(?:\.\d+)?)\s*(cm|mm|m)\b'
    for match in re.finditer(unit_pattern, text_lower):
        val = match.group(1)
        # Check if this value is already part of a list
        is_duplicate = any(val in dim for dim in found_values if 'x' in dim)
        if not is_duplicate:
            found_values.append(val)

    # --- CLEANUP  ---
    seen = set()
    unique_final = []
    for v in found_values:
        if v not in seen:
            seen.add(v)
            unique_final.append(v)

    return unique_final

# Helper function to remove accents and standrize the text
def remove_accents(input_str):
    if not input_str:
        return ""
    org_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in org_form if not unicodedata.combining(c)])

def extract_colors(text):
    """
    Extracts colors using a comprehensive French/English dictionary 
    """
    if pd.isna(text) or text == "":
        return []

    # 1. CLEANING: Lowercase, remove accents, replace separators
    # "Brun-Chocolat" becomes "brun chocolat" so regex \b works
    clean_text = remove_accents(str(text)).lower()
    clean_text = clean_text.replace('-', ' ').replace('/', ' ').replace(':', ' ').replace('+', ' ')
    found_colors = []

    # 2. THE MEGA-MAP (Grouped by Color Family)
    # The Key is the "Standard Name"and The Value is the Regex matching all variations.
    color_map = {
        # --- BROWNS ---
        'marron': r'\bmarron(?:s)?\b|\bbrun(?:e|s|es)?\b|\bbrown\b|\bchocolat\b|\bcacao\b|\bcafe\b|\btabac\b|\bcoffee\b|\bnoisette\b|\bchatain\b',
        'camel': r'\bcamel\b|\bcognac\b|\bcaramel\b|\bocre\b|\bbrique\b|\brouille\b|\brust\b|\bterracotta\b|\bhale\b',
        'bois': r'\bbois\b|\bwood\b|\bchene\b|\boak\b|\bhetre\b|\bnoyer\b|\bwalnut\b|\bteck\b|\bacacia\b|\bpin\b|\bbambou\b|\brotin\b|\bosier\b|\bmerisier\b|\bolivier\b|\bpalissandre\b|\bhevea\b',

        # --- WHITES / NEUTRALS ---
        'blanc': r'\bblanc(?:he|s|hes)?\b|\bwhite\b|\bneige\b',
        'beige': r'\bbeige(?:s)?\b|\bsable\b|\bsand\b|\bcream\b|\bcreme\b|\bchampagne\b',
        'naturel': r'\bnatur(?:el|elle|els|elles)\b|\bneutre\b|\braw\b|\bbrut\b',
        'ivoire': r'\bivoire\b|\bivory\b|\becru\b|\bcoquille\b|\bvanille\b',
        'transparent': r'\btransparent(?:e|s|es)?\b|\bcristal\b|\bclear\b|\bverre\b',

        # --- GRAYS / BLACKS ---
        'noir': r'\bnoir(?:e|s|es)?\b|\bblack\b|\bcarbon(?:e)?\b|\bencre\b|\bebene\b',
        'gris': r'\bgris(?:e|es)?\b|\bgrey\b|\bgray\b|\banthracite\b|\bbeton\b|\bconcrete\b|\bardois(?:e)?\b|\bmetal\b|\bacier\b|\bplomb\b|\btain\b|\bgalet\b|\bsouris\b',
        'taupe': r'\btaupe\b|\bgrege\b',

        # --- BLUES ---
        'bleu': r'\bbleu(?:e|s|es)?\b|\bblue\b|\bmarine\b|\bnavy\b|\bazur\b|\bcyan\b|\bturquoise\b|\bindigo\b|\bdenim\b|\bpetrole\b|\bcanard\b|\bnuit\b|\bciel\b|\bsaphir\b|\broi\b',

        # --- GREENS ---
        'vert': r'\bvert(?:e|s|es)?\b|\bgreen\b|\bkaki\b|\bkhaki\b|\bolive\b|\bsauge\b|\bforet\b|\bforest\b|\bmenthe\b|\bmint\b|\banis\b|\bsapin\b|\bmeraude\b|\btilleul\b|\bpistache\b',

        # --- REDS / PINKS ---
        'rouge': r'\brouge(?:s)?\b|\bred\b|\bbordeaux\b|\bcerise\b|\bcherry\b|\bgrenat\b|\bburgundy\b|\brubis\b|\bcarmin\b|\btomate\b|\bcoquelicot\b',
        'rose': r'\brose(?:s)?\b|\bpink\b|\bpoudr(?:e)?\b|\bfushia\b|\bfuchsia\b|\bcorail\b|\bcoral\b|\bsaumon\b|\bpeche\b|\bmagenta\b|\bframboise\b',
        'violet': r'\bviolet(?:te|s|tes)?\b|\bpurple\b|\bprune\b|\bplum\b|\blilas\b|\baubergine\b|\bmauve\b|\bparme\b|\bvioline\b',

        # --- YELLOWS / ORANGES ---
        'jaune': r'\bjaune(?:s)?\b|\byellow\b|\bmoutarde\b|\bmustard\b|\bcitron\b|\blemon\b|\bsoleil\b|\bpaille\b|\bcurry\b',
        'orange': r'\borange(?:s)?\b|\bmandarine\b|\btangerine\b|\babricot\b',

        # --- METALLICS ---
        'doré': r'\bdor(?:e|ee|es|ees)?\b|\bgold(?:en)?\b|\blaiton\b|\bbrass\b',
        'argenté': r'\bargent(?:e|ee|es|ees)?\b|\bsilver\b|\bchrom(?:e|ee|es|ees)?\b|\binox\b',
        'cuivre': r'\bcuivr(?:e|ee|es|ees)?\b|\bcopper\b|\bbronze\b',

        # --- PATTERNS ---
        'multicolore': r'\bmulticolore\b|\bmulticouleur\b|\bcolor(?:e|ee|es|ees)\b|\bimprim(?:e)?\b|\bmotif(?:s)?\b|\brayur(?:e|es)\b|\bpatchwork\b',
    }

    # 3. SEARCH THE MAP
    for standard_name, pattern in color_map.items():
        if re.search(pattern, clean_text):
            found_colors.append(standard_name)

    # 4. SAFETY NET (Context Extraction)
    # If we haven't found a color yet, look for explicit labels like "Couleur : X"
    # This catches special colors we didn't list (e.g., "Couleur : Whisky")
    if not found_colors:
        context_pattern = r'(?:couleur|coloris|teinte|finition|color)\s*(?:\:)?\s*([a-z]{3,15})'
        match = re.search(context_pattern, clean_text)
        if match:
            extracted_word = match.group(1)
            if extracted_word not in ['le', 'la', 'de', 'du', 'des', 'et', 'pour', 'avec']:
                found_colors.append(extracted_word)

    # Remove duplicates
    return list(set(found_colors))

def extract_features(df, description_col='Description'):
    """
    Extracts dimensions and colors, handles List returns, and calculates correct stats.
    """
    logging.info("=== EXTRACTION DES DIMENSIONS ET COULEURS ===")
    
    # Work on a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # 1. Apply the functions (Ensure you have imported extract_all_dimensions and extract_colors)
    logging.info(" Extraction des dimensions en cours...")
    df_copy['dimensions_extraites'] = df_copy[description_col].apply(extract_all_dimensions)
    
    logging.info("Extraction des couleurs en cours...")
    df_copy['couleurs_extraites'] = df_copy[description_col].apply(extract_colors)
    
    # 2. Calculate Statistics
    # Since the columns contain lists [], .notna() doesn't work. 
    # We check if the list has a length > 0.
    dim_count = df_copy['dimensions_extraites'].apply(lambda x: len(x) > 0).sum()
    color_count = df_copy['couleurs_extraites'].apply(lambda x: len(x) > 0).sum()
    
    total = len(df)
    
    # 3. Print Results
    logging.info(f"Dimensions trouvées : {dim_count} produits ({dim_count/total*100:.1f}%)")
    logging.info(f"Couleurs trouvées   : {color_count} produits ({color_count/total*100:.1f}%)")

    return df_copy


if __name__ == "__main__":

    filepath = r"C:\Users\yo-un\OneDrive\Desktop\Internship_Use_Case\Ecommerce_Data\Ecommerce_sales.xlsb"
    
    # 1. Data Loading
    df = load_data(filepath)
    
    # 2. Analyze Categories
    Analyse_Categories = analyze_categories(df, category_col="Nature")
    
    # 3. Detect & Fix Categories 
    # This creates 'Cat_Verified_df', which has the corrected 'Nature' column
    Cat_Verified_df, Mis_Categorized_Products = detect_miscategorized_products(
        df, 
        nature_col='Nature', 
        description_col='Libellé produit', 
        threshold=0.3
    )
    
    # 4. Extract Features on the CORRECTED Dataset
    # We pass 'Cat_Verified_df' here, not 'df'
    Final_Dataset = extract_features(Cat_Verified_df, description_col='Libellé produit')

    # 5. Convert Lists to Strings for better readability in CSV file
    Final_Dataset['Product_Dimensions'] = Final_Dataset['dimensions_extraites'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
    Final_Dataset['Product_Couleurs'] = Final_Dataset['couleurs_extraites'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')    
    Final_Dataset.drop(columns=['dimensions_extraites', 'couleurs_extraites'], inplace=True)
    # Save to CSV
    output_path = r"C:\Users\yo-un\OneDrive\Desktop\Internship_Use_Case\Ecommerce_Data\Exported_Ecommerce_sales.csv"
    Final_Dataset.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
    logging.info(f"Dataset saved to {output_path}")