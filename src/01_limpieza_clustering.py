import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import sys
from pathlib import Path

# --- BLOQUE DE CONEXIÓN E INFRAESTRUCTURA ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

# Importaciones dinámicas desde config.py para evitar rutas quemadas (Hardcoded paths)
from config import PATH_CORPUS_CRUDO, PATH_CORPUS_AGRUPADO, PALABRAS_IRRELEVANTES

# ==========================================
# 1. PARÁMETROS DEL MODELO
# ==========================================
NUM_CLUSTERS = 15
# PASO B: Llena esta lista después de ver de qué trata cada uno de los 15 grupos
GRUPOS_A_ELIMINAR = [1, 2, 4, 6, 8, 9, 10, 11, 12, 14] 

# Optimización AHEAD OF TIME (Reduce el tiempo de ejecución dramáticamente)
REGEX_PALABRAS = re.compile(r'\b[a-záéíóúñü]+\b')
# Búsqueda O(1) para filtrado ultrarrápido
SET_IRRELEVANTES = set(PALABRAS_IRRELEVANTES)

# ==========================================
# 2. FUNCIONES DE ANÁLISIS OPTIMIZADAS
# ==========================================

def obtener_frecuencias(lista_textos):
    """Cuenta palabras usando iteradores para proteger la memoria RAM."""
    contador = Counter()
    for texto in lista_textos:
        # Se asegura de procesar solo strings válidos
        if isinstance(texto, str):
            palabras = REGEX_PALABRAS.findall(texto.lower())
            contador.update(palabras)
    return contador

def limpiar_texto(texto):
    """Quita palabras irrelevantes de un texto individual de forma vectorizada."""
    if not isinstance(texto, str):
        return ""
    palabras = REGEX_PALABRAS.findall(texto.lower())
    palabras_limpias = [p for p in palabras if p not in SET_IRRELEVANTES]
    return " ".join(palabras_limpias)

# ==========================================
# 3. FLUJO PRINCIPAL
# ==========================================

def analizar_corpus():
    print("📂 Cargando el corpus de letras crudo...")
    try:
        df = pd.read_csv(PATH_CORPUS_CRUDO)
        df = df.dropna(subset=['lyrics']).copy() 
    except FileNotFoundError:
        print(f"❌ Error crítico: No se encontró el archivo en la ruta:\n{PATH_CORPUS_CRUDO}")
        return

    # ---------------------------------------------------------
    # FASE 1: Frecuencias crudas y Validación de Stop-Words
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 FASE 1: FRECUENCIAS GENERALES Y DIAGNÓSTICO")
    print("="*50)
    
    frecuencias_crudas = obtener_frecuencias(df['lyrics'])
    
    if not PALABRAS_IRRELEVANTES:
        print("Top 30 palabras más usadas en el corpus crudo:")
        for palabra, cantidad in frecuencias_crudas.most_common(30):
            print(f" - {palabra}: {cantidad}")
        print("\n🛑 PAUSA ESTRATÉGICA: Configura la variable 'PALABRAS_IRRELEVANTES'")
        print("en tu archivo config.py antes de continuar el pipeline.")
        return
    else:
        print("🛑 PALABRAS RELEVANTES (Top 100 excluyendo Stop-Words):")
        contador_mostradas = 0
        for palabra, cantidad in frecuencias_crudas.most_common():
            if palabra not in SET_IRRELEVANTES:
                print(f" - {palabra}: {cantidad}")
                contador_mostradas += 1
            if contador_mostradas == 100:
                break

    # ---------------------------------------------------------
    # FASE 2: Limpieza y Clustering (Cercanía semántica)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"🧹 FASE 2: EXTRACCIÓN DE CARACTERÍSTICAS Y CLUSTERING ({NUM_CLUSTERS} Grupos)")
    print("="*50)
    
    print("⏳ Aplicando limpieza de texto a nivel corpus...")
    df['lyrics_limpias'] = df['lyrics'].apply(limpiar_texto)
    
    print("🧠 Calculando tensores espaciales (TF-IDF)...")
    vectorizador = TfidfVectorizer(max_features=5000)
    matriz_tfidf = vectorizador.fit_transform(df['lyrics_limpias'])
    
    print("🤖 Entrenando modelo no supervisado (K-Means)...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
    df['subgenero_ia'] = kmeans.fit_predict(matriz_tfidf)
    
    # Perfilado de centroides
    terminos = vectorizador.get_feature_names_out()
    centroides = kmeans.cluster_centers_.argsort()[:, ::-1]
    
    print("\n🔍 PERFILADO DE CLÚSTERES:")
    for i in range(NUM_CLUSTERS):
        palabras_clave = [terminos[ind] for ind in centroides[i, :7]]
        total_canciones = (df['subgenero_ia'] == i).sum()
        estado = "❌ DESCARTADO" if i in GRUPOS_A_ELIMINAR else "✅ CONSERVADO"
        print(f"Grupo {i:02d} [{estado}] ({total_canciones} tracks) -> Core: {', '.join(palabras_clave)}")

    if not GRUPOS_A_ELIMINAR:
        print("\n🛑 PAUSA ESTRATÉGICA: Analiza los perfiles de los clústeres.")
        print("Define los grupos ruidosos en la variable 'GRUPOS_A_ELIMINAR' y re-ejecuta.")
        return

    # ---------------------------------------------------------
    # FASE 3: Eliminación, Recálculo y Persistencia
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("✂️ FASE 3: RECORTE Y SERIALIZACIÓN FINAL")
    print("="*50)
    
    df_filtrado = df[~df['subgenero_ia'].isin(GRUPOS_A_ELIMINAR)].copy()
    canciones_borradas = len(df) - len(df_filtrado)
    
    print(f"🗑️ Ruido eliminado: {canciones_borradas} canciones.")
    print(f"✅ Corpus efectivo: {len(df_filtrado)} canciones.\n")
    
    print("Top 100 palabras definitorias del CORPUS EFECTIVO:")
    frecuencias_finales = obtener_frecuencias(df_filtrado['lyrics_limpias'])
    for palabra, cantidad in frecuencias_finales.most_common(100):
        print(f" - {palabra}: {cantidad}")
        
    # Escritura final en disco
    df_filtrado.to_csv(PATH_CORPUS_AGRUPADO, index=False, encoding='utf-8-sig')
    print(f"\n💾 [SISTEMA] Corpus efectivo serializado en:\n{PATH_CORPUS_AGRUPADO}")

if __name__ == "__main__":
    analizar_corpus()