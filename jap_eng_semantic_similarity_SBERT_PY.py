# ==================================================================================================================================================================

# Author                    : Dr. Marcos H. Cárdenas Mancilla
# E-mail                    : marcos.cardenas.m@usach.cl
# Date of creation          : 2025-01-11
# Licence                   : AGPL V3
# Copyright (c) 2025 Marcos H. Cárdenas Mancilla.

# ==================================================================================================================================================================

# Descripción JAP_ENG_SEMANTIC_SIMILARITY_SBERT_PY:
# Este código Python analiza la similitud semántica entre pares de textos en japonés e inglés para validar los resultados del análisis comparativo de la estructura 
# argumental de ambas lenguas.
# 
# Características:
# 1. importa de datos de una matriz de fichas analíticas en formato .csv, combinando las columnas de "Proceso Verbal" con otras para crear textos conjuntos en JP-EN.
# 2. implementa el modelo de Sentence-BERT, 'paraphrase-multilingual-MiniLM-L12-v2', para generar embeddings y calcular la similitud coseno entre 
# las representaciones vectoriales de cada par. 
# 3. categoriza las similitudes obtenidas: Baja Similitud (0,0 – 0,5), Similitud Moderada (0,5 – 0,8) y Alta Similitud (0,8 – 1,0). 
# Los resultados se evalúan mediante análisis descriptivos e inferenciales como la prueba de Kruskal-Wallis.
# 4. aplica pruebas estadísticas (Shapiro-Wilk y Kruskal-Wallis) para analizar la distribución y diferencias significativas entre estas categorías.
# 5. exporta los resultados a archivos .csv.
# 
# El objetivo principal es evaluar,  a través del cáculo de la similitud semántica, la calidad del análisis comparativo de las estructuras argumentales y 
# morfosintácticas del japonés e inglés. Un alto puntaje de similitud indica que la traducción al inglés mantiene el significado del original, 
# mientras que un puntaje bajo puede señalar problemas en la comparación o diferencias gramaticales que impactan el significado.

# ==================================================================================================================================================================

# Cargar librerías necesarias
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from scipy.stats import kruskal, shapiro
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Cargar y limpiar la base de datos
file_path = r'matriz_fichas_analíticas.csv'  # Cambiar por la ruta correcta
matriz_fichas = pd.read_csv(file_path, delimiter=';')

# Seleccionar y renombrar columnas relevantes
matriz_fichas_renombrada = matriz_fichas.rename(columns={
    'Unnamed: 2': 'Proceso Verbal',
    'Unnamed: 4': 'Argumento Asociado',
    '[a]': 'Identificador Principal'
})

# Filtrar columnas útiles y eliminar valores nulos
datos_relevantes = matriz_fichas_renombrada[['Identificador Principal', 'Proceso Verbal', 'Argumento Asociado']].dropna()

# Crear texto combinado para japonés e inglés
datos_relevantes['Texto Japonés'] = datos_relevantes['Proceso Verbal'] + " " + datos_relevantes['Argumento Asociado']
datos_relevantes['Texto Inglés'] = datos_relevantes['Proceso Verbal'] + " " + datos_relevantes['Argumento Asociado']

# Cargar el modelo multilingüe preentrenado de Sentence-BERT
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Función para calcular similitud semántica
def calcular_similitud_sbert(row):
    try:
        jap_embedding = model.encode(row['Texto Japonés'], convert_to_tensor=True)
        eng_embedding = model.encode(row['Texto Inglés'], convert_to_tensor=True)
        return util.pytorch_cos_sim(jap_embedding, eng_embedding).item()
    except Exception:
        return None

# Aplicar el cálculo de similitud
datos_relevantes['Similitud Semántica'] = datos_relevantes.apply(calcular_similitud_sbert, axis=1)

# Exportar los resultados a un archivo CSV
datos_relevantes.to_csv(r'comparaciones_jap_ingles_sbert.csv', index=False)

# Mostrar los resultados
print("Resultados de Similitud Semántica:")
print(datos_relevantes[['Identificador Principal', 'Similitud Semántica']].head())

# Cargar el archivo inicial
file_path = r'df_filtered.csv'  # Cambiar por la ruta correcta
df_filtered = pd.read_csv(file_path)

# Paso 1: Archivo Cargado - Resumen
print("Archivo cargado: Resumen")
print(df_filtered.head())

# Paso 2: Validación de Traducciones Jap-Eng
print("\nValidación de Traducciones Jap-Eng")
subconjunto_validacion = df_filtered[['jap', 'eng']].sample(n=5, random_state=42)
print(subconjunto_validacion)

# Paso 3: Comparación Con Similitudes Semánticas
df_filtered['Texto Japonés'] = df_filtered['jap'] + " " + df_filtered['subj'].fillna('') + " " + df_filtered['obj'].fillna('')
df_filtered['Texto Inglés'] = df_filtered['eng'] + " " + df_filtered['subj'].fillna('') + " " + df_filtered['obj'].fillna('')

# Calcular similitudes semánticas usando TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_filtered['Texto Japonés'] + df_filtered['Texto Inglés'])
similitudes = cosine_similarity(tfidf_matrix[:len(df_filtered)])

# Asignar las similitudes calculadas al DataFrame
df_filtered['Similitud Semántica'] = [similitudes[i, i] for i in range(len(similitudes))]

# Paso 4: Validación de Traducciones y Similitudes
subconjunto_similitudes = df_filtered[['Texto Japonés', 'Texto Inglés', 'Similitud Semántica']].sample(n=5, random_state=42)
print("\nValidación de Traducciones y Similitudes")
print(subconjunto_similitudes)

# Paso 5: Procesos Verbales Limpios
procesos_limpios = df_filtered.dropna(subset=['Texto Japonés', 'Texto Inglés'])

# Paso 6: Resultados Finales Con Filas Completas
vectorizer_final = TfidfVectorizer()
tfidf_matrix_final = vectorizer_final.fit_transform(procesos_limpios['Texto Japonés'] + " " + procesos_limpios['Texto Inglés'])
similitudes_final = cosine_similarity(tfidf_matrix_final[:len(procesos_limpios)])
procesos_limpios['Similitud Semántica'] = [similitudes_final[i, i] for i in range(len(similitudes_final))]

# Exportar resultados finales
output_file_final = r'procesos_verbales_similitud_final.csv'
procesos_limpios.to_csv(output_file_final, index=False)

# Paso 7: Distribución Final De Similitudes Semánticas
plt.hist(procesos_limpios['Similitud Semántica'], bins=10, edgecolor='k')
plt.title('Distribución Final de Similitudes Semánticas')
plt.xlabel('Similitud Semántica')
plt.ylabel('Frecuencia')
plt.show()

# Paso 8: Revisión Final De Resultados
print("\nRevisión Final de Resultados")
subconjunto_final = procesos_limpios[['Texto Japonés', 'Texto Inglés', 'Similitud Semántica']].sample(n=5, random_state=42)
print(subconjunto_final)

# Paso 9: Validación Final de los Resultados
print("\nValidación Final de los Resultados")
if all(procesos_limpios['Similitud Semántica'] == 1.0):
    print("Todas las similitudes semánticas reflejan correspondencia perfecta.")
else:
    print("Existen similitudes semánticas por debajo de 1.0; se recomienda revisión adicional.")

# Paso 10: Cargar como input el resultado de la ejecución del Paso 6.
file_path = r'procesos_verbales_similitud_final.csv'  # Cambiar por la ruta correcta
procesos_completados = pd.read_csv(file_path)

# Generar embeddings y calcular similitudes
def calcular_similitud(row):
    try:
        jap_embedding = model.encode(row['Texto Japonés'], convert_to_tensor=False)
        eng_embedding = model.encode(row['Texto Inglés'], convert_to_tensor=False)
        return util.cos_sim(jap_embedding, eng_embedding).item()
    except Exception as e:
        return None

# Paso 11: Aplicar la función de similitud
procesos_completados['Similitud Semántica (BERT)'] = procesos_completados.apply(calcular_similitud, axis=1)

# Paso 12: Clasificar según similitudes calculadas
procesos_completados['Clasificación Morfosintáctica (BERT)'] = pd.cut(
    procesos_completados['Similitud Semántica (BERT)'],
    bins=[0, 0.5, 0.8, 1.0],
    labels=['Baja Similitud', 'Similitud Moderada', 'Alta Similitud'],
    include_lowest=True
)

# Paso 13: Exportar los resultados ajustados
output_file_bert = r'procesos_verbales_similitud_bert.csv'
procesos_completados.to_csv(output_file_bert, index=False)

print(f"Resultados ajustados con Sentence-BERT exportados a: {output_file_bert}")

# Paso 14: Agrupar datos por categoría
baja_similitud = procesos_completados[procesos_completados['Clasificación Morfosintáctica (BERT)'] == 'Baja Similitud']['Similitud Semántica (BERT)']
moderada_similitud = procesos_completados[procesos_completados['Clasificación Morfosintáctica (BERT)'] == 'Similitud Moderada']['Similitud Semántica (BERT)']
alta_similitud = procesos_completados[procesos_completados['Clasificación Morfosintáctica (BERT)'] == 'Alta Similitud']['Similitud Semántica (BERT)']

# Paso 15: Verificar la normalidad de los datos
print("Prueba de normalidad (Shapiro-Wilk):")
print(f"Baja Similitud: {shapiro(baja_similitud)}")
print(f"Similitud Moderada: {shapiro(moderada_similitud)}")
print(f"Alta Similitud: {shapiro(alta_similitud)}")

# Paso 16: Aplicar la prueba de Kruskal-Wallis
result = kruskal(baja_similitud, moderada_similitud, alta_similitud)

print(f"Estadístico de Prueba: {result.statistic}, Valor p: {result.pvalue}")
