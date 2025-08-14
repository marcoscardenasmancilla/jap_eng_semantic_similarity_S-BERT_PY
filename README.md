# jap_eng_semantic_similarity_S-BERT_PY

# Author                    : Dr. Marcos H. Cárdenas Mancilla
# E-mail                    : marcoscardenasmancilla@gmail.com
# Date of creation          : 2025-01-11
# Licence                   : AGPL V3
# Copyright (c) 2025 Marcos H. Cárdenas Mancilla.

# Descripción JAP_ENG_SEMANTIC_SIMILARITY_SBERT_PY:
# Este código Python analiza la similitud semántica entre pares de textos en japonés e inglés para validar los resultados del análisis comparativo de la estructura argumental de ambas lenguas.

# Características:
# 1. importa datos lingüísticos de una matriz de fichas analíticas en formato .csv. Combina las columnas de "Proceso Verbal" con aquellas vinculadas a los "Argumentos" y "Contextos" que permiten crear textos conjuntos en JP-EN.
# 2. implementa el modelo de Sentence-BERT, 'paraphrase-multilingual-MiniLM-L12-v2', para generar embeddings y calcular la similitud coseno entre las representaciones vectoriales de cada par. 
# 3. categoriza las similitudes obtenidas (alta, moderada, baja).
# 4. aplica pruebas estadísticas (Shapiro-Wilk y Kruskal-Wallis) para analizar la distribución y diferencias significativas entre estas categorías.
# 5. exporta los resultados a archivos .csv.

# El objetivo principal es evaluar,  a través del cálculo de la similitud semántica, la calidad del análisis comparativo de las estructuras argumentales y  morfosintácticas del japonés e inglés. Un alto puntaje de similitud indica que la traducción al inglés mantiene el significado del original, mientras que un puntaje bajo puede señalar problemas en la comparación o diferencias gramaticales que impactan el significado.
