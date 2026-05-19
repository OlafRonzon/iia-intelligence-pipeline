import pandas as pd
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# --- CONEXIÓN ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    from config import DIR_INTERMEDIATE, DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    PATH_PISCINA = DIR_INTERMEDIATE / "03_piscina_versos_restantes.csv"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit()

def generar_boosters_vectoriales(n_versos=100):
    print("🚀 Cargando modelo SOTA para minería semántica...")
    # Usamos un modelo multilingüe de alto rendimiento
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # 1. Cargar la piscina
    df_piscina = pd.read_csv(PATH_PISCINA)
    print(f"🌊 Piscina cargada: {len(df_piscina)} versos.")
    
    # 2. Vectorizar la piscina (Esto puede tomar un minuto)
    print("🧠 Calculando vectores de toda la piscina (esto es bleeding edge)...")
    embeddings_piscina = model.encode(df_piscina['verso_texto'].tolist(), show_progress_bar=True)
    
    for dimension, palabras in DICCIONARIO_PENTADIMENSIONAL.items():
        # Blindaje corregido: Busca el nombre exacto del archivo que este script genera
        archivo_esperado = DIR_VALIDATION / f"5_booster_{dimension}_vector.csv"
        
        if archivo_esperado.exists():
            print(f"⏭️ Booster Vectorial para {dimension} ya existe. Saltando...")
            continue
            
        print(f"🎯 Minando semánticamente la dimensión: {dimension}...")
        
        # 3. Crear el "Vector Ideal" de la dimensión
        # Promediamos los vectores de las palabras clave para crear un "centroide"
        vector_ideal = model.encode(palabras).mean(axis=0)
        
        # 4. Calcular similitud del coseno
        cos_scores = util.cos_sim(vector_ideal, embeddings_piscina)[0]
        
        # 5. Obtener los versos más similares (TOP N)
        top_results = np.argpartition(-cos_scores, range(n_versos))[:n_versos]
        
        df_booster = df_piscina.iloc[top_results].copy()
        
        # 6. Preparar para calificación manual
        columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
        for col in columnas_val:
            df_booster[col] = ""
            
        columnas_finales = ['artist', 'song', 'year', 'verso_texto'] + columnas_val
        df_booster = df_booster.reindex(columns=columnas_finales)
        
        # 7. Guardar con marca vectorial
        nombre_archivo = f"5_booster_{dimension}_vector.csv"
        df_booster.to_csv(DIR_VALIDATION / nombre_archivo, index=False, encoding='utf-8-sig')
        print(f"✅ Booster vectorial generado: {nombre_archivo}")

if __name__ == "__main__":
    generar_boosters_vectoriales(n_versos=100)