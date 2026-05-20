import pandas as pd
import numpy as np
import sys
import os
import pickle
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    dir_src = Path(os.getcwd()) / "src"

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_INTERMEDIATE, DIR_PROCESSED, PATH_CORPUS_FILTRADO_CLUSTERS, DICCIONARIO_PENTADIMENSIONAL
    
    # Input: Tu base masiva de versos
    PATH_INPUT_MASIVO = PATH_CORPUS_FILTRADO_CLUSTERS 
    
    # El cerebro equilibrado
    PATH_MODELS_IN = DIR_INTERMEDIATE / "multioutput_xgb_models_balanced.pkl" 
    
    # Output: El resultado final de tu tesis
    PATH_OUTPUT_CSV = DIR_PROCESSED / "6_corpus_masivo_etiquetado_ia.csv"
    PATH_OUTPUT_PARQUET = DIR_PROCESSED / "6_corpus_masivo_etiquetado_ia.parquet"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

MODELO_SOTA = 'BAAI/bge-m3'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256 # Tamaño de lote óptimo para GPU T4 en Colab

def despliegue_masivo():
    print("🚀 Iniciando Fase 4: Despliegue Masivo de Inferencia (Módulo 07)")
    
    # 1. Validaciones
    if not PATH_INPUT_MASIVO.exists():
        print(f"❌ No se encontró la base masiva en: {PATH_INPUT_MASIVO}")
        sys.exit(1)
    if not PATH_MODELS_IN.exists():
        print(f"❌ No se encontró el cerebro artificial en: {PATH_MODELS_IN}")
        sys.exit(1)
        
    # 2. Carga de Datos
    print("📂 Cargando la base de datos masiva...")
    df_masivo = pd.read_csv(PATH_INPUT_MASIVO)
    
    # Asegurarnos de no procesar filas vacías
    df_masivo = df_masivo.dropna(subset=['verso_texto'])
    total_versos = len(df_masivo)
    print(f"📊 Versos totales a procesar: {total_versos}")
    
    # 3. Vectorización (El cuello de botella computacional)
    print(f"⚡ Iniciando Vectorización en [{DEVICE.upper()}] con {MODELO_SOTA}...")
    encoder = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    
    # Extraemos la lista de textos
    textos = df_masivo['verso_texto'].tolist()
    
    # Convertimos a vectores (Esto tomará tiempo, show_progress_bar dibujará una barra en Colab)
    embeddings = encoder.encode(textos, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE, convert_to_numpy=True)
    print("✅ Vectorización masiva completada.")
    
    # 4. Clasificación Multi-Etiqueta
    print("🧠 Despertando al Cerebro Artificial (Modelos XGBoost)...")
    with open(PATH_MODELS_IN, 'rb') as f:
        modelos = pickle.load(f)
        
    dimensiones = list(DICCIONARIO_PENTADIMENSIONAL.keys())
    
    for dim in dimensiones:
        print(f"   ► Evaluando dimensión: {dim}...")
        xgb_model = modelos[dim]
        # Predicción masiva instantánea
        predicciones = xgb_model.predict(embeddings)
        # Asignamos la nueva columna al DataFrame
        df_masivo[f'intensidad_{dim}'] = predicciones
        
    # 5. Exportación
    print("💾 Guardando el Corpus Masivo Etiquetado...")
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Guardamos en CSV para que puedas leerlo en Excel
    df_masivo.to_csv(PATH_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    # Guardamos en Parquet (Ideal para cruces de datos posteriores sin perder tipos)
    df_masivo.to_parquet(PATH_OUTPUT_PARQUET, index=False)
    
    print(f"🎉 ¡Fase 4 Completada!")
    print(f"   📄 CSV: {PATH_OUTPUT_CSV}")
    print(f"   📦 Parquet: {PATH_OUTPUT_PARQUET}")

if __name__ == "__main__":
    despliegue_masivo()