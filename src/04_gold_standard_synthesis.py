import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Configuración de rutas (Ajusta según tu entorno)
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    dir_src = Path(os.getcwd()) / "src"

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_VALIDATION, DIR_INTERMEDIATE, DICCIONARIO_PENTADIMENSIONAL, PATH_GOLD_STANDARD_VECTORS, PATH_GOLD_STANDARD_LABELS, PATH_STATS
    PATH_GOLD_STANDARD = PATH_GOLD_STANDARD_VECTORS
    PATH_GOLD_LABELS = PATH_GOLD_STANDARD_LABELS
    
    DIR_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

MODELO_SOTA = 'BAAI/bge-m3'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def procesar_etl_gold_standard():
    print("🚀 Iniciando Fase 1: Síntesis del Gold Standard")
    archivos_csv = list(DIR_VALIDATION.glob("5_booster_*.csv"))
    
    if not archivos_csv:
        print("❌ No se encontraron archivos en la carpeta de validación.")
        sys.exit(1)
        
    lista_dfs = []
    conteo_regex = 0
    conteo_kmeans = 0
    
    # 1. Ingestión y Etiquetado de Origen
    for archivo in archivos_csv:
        df = pd.read_csv(archivo)
        if 'v2' in archivo.name:
            df['origen'] = 'K-Means'
            conteo_kmeans += len(df)
        else:
            df['origen'] = 'Regex'
            conteo_regex += len(df)
        lista_dfs.append(df)
        
    df_maestro = pd.concat(lista_dfs, ignore_index=True)
    total_bruto = len(df_maestro)
    
    # 2. Deduplicación (Retener Intensidad Máxima)
    columnas_val = [f"val_{dim}" for dim in DICCIONARIO_PENTADIMENSIONAL.keys()]
    for col in columnas_val:
        if col in df_maestro.columns:
            df_maestro[col] = pd.to_numeric(df_maestro[col], errors='coerce').fillna(0)
        else:
            df_maestro[col] = 0
            
    df_maestro = df_maestro.dropna(subset=['verso_texto'])
    
    # Agrupamos reteniendo el clímax de los valores y el primer año
    diccionario_agrupacion = {col: 'max' for col in columnas_val}
    diccionario_agrupacion['year'] = 'first'
    
    df_deduplicado = df_maestro.groupby('verso_texto', as_index=False).agg(diccionario_agrupacion)
    total_deduplicado = len(df_deduplicado)
    print(f"📊 Deduplicación: {total_bruto} brutos -> {total_deduplicado} únicos.")
    
    return df_deduplicado, conteo_regex, conteo_kmeans

def aplicar_sobremuestreo_sintetico(df):
    print(f"\n📦 Cargando Modelo Lingüístico para Vectorización: {MODELO_SOTA}...")
    model = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    
    textos = df['verso_texto'].tolist()
    columnas_val = [f"val_{dim}" for dim in DICCIONARIO_PENTADIMENSIONAL.keys()]
    
    # Vectorizamos los textos reales
    print("🧠 Vectorizando base deduplicada...")
    embeddings = model.encode(textos, show_progress_bar=True, device=DEVICE, convert_to_numpy=True)
    
    # Preparar el Target (La Década)
    df['decada'] = (pd.to_numeric(df['year'], errors='coerce').fillna(2000) // 10) * 10
    y_decadas = df['decada'].astype(int).values
    
    # Unimos los embeddings con las etiquetas de intensidad (para que SMOTE las interpole también)
    intensidades = df[columnas_val].values
    X_combinado = np.hstack((embeddings, intensidades))
    
    # Configuración dinámica de SMOTE para evitar fallos si una década tiene pocos versos
    conteo_clases = Counter(y_decadas)
    min_samples = min(conteo_clases.values())
    k_vecinos = min(5, min_samples - 1)
    
    print("\n⚖️ Aplicando SMOTE (Anti-Teleología)...")
    if k_vecinos > 0:
        smote = SMOTE(k_neighbors=k_vecinos, random_state=42)
        X_sintetico, y_sintetico_decadas = smote.fit_resample(X_combinado, y_decadas)
    else:
        print("⚠️ Hay décadas con solo 1 verso. SMOTE requiere al menos 2. Se omitirá el sobremuestreo.")
        X_sintetico, y_sintetico_decadas = X_combinado, y_decadas
        
    total_final = len(X_sintetico)
    
    # Separar los embeddings de las intensidades sintéticas
    dim_embedding = embeddings.shape[1]
    embeddings_finales = X_sintetico[:, :dim_embedding]
    
    # Redondeamos las intensidades sintéticas para que vuelvan a ser 0, 1, 2 o 3
    intensidades_finales = np.clip(np.round(X_sintetico[:, dim_embedding:]), 0, 3)
    
    print(f"✅ Balanceo completado: {len(embeddings)} reales -> {total_final} totales (reales + sintéticos).")
    
    # Guardar matrices para la Fase 3
    np.save(PATH_GOLD_STANDARD, embeddings_finales)
    np.save(PATH_GOLD_LABELS, intensidades_finales)
    
    return total_final

def main():
    df_deduplicado, conteo_regex, conteo_kmeans = procesar_etl_gold_standard()
    total_final = aplicar_sobremuestreo_sintetico(df_deduplicado)
    
    # Guardar estadísticas para el módulo 04a (Sankey)
    stats = {
        "regex_in": conteo_regex,
        "kmeans_in": conteo_kmeans,
        "deduplicado": len(df_deduplicado),
        "smote_out": total_final,
        "sinteticos_creados": total_final - len(df_deduplicado)
    }
    
    with open(PATH_STATS, 'w') as f:
        json.dump(stats, f)
        
    print(f"\n🎉 Fase 1 completada. Stats guardadas en: {PATH_STATS}")

if __name__ == "__main__":
    main()