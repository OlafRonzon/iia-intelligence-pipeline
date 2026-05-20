import pandas as pd
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# RUTAS DIRECTAS A TU GOOGLE DRIVE
# ---------------------------------------------------------
# Estas rutas asumen que ya montaste tu Drive en Colab
CARPETA_DRIVE = "/content/drive/MyDrive/Datos_Corridos"

PATH_INPUT_MASIVO = f"{CARPETA_DRIVE}/03_piscina_versos_restantes.csv"
PATH_MODELS_IN = f"{CARPETA_DRIVE}/multioutput_xgb_models_balanced.pkl" 

# El resultado se guardará directo en tu Drive para que no lo pierdas
PATH_OUTPUT_CSV = f"{CARPETA_DRIVE}/04_corpus_masivo_etiquetado.csv"

# Diccionario de las dimensiones que auditará
DICCIONARIO_PENTADIMENSIONAL = {'AR': 1, 'MO': 1, 'DI': 1, 'PO': 1, 'NA': 1, 'GO': 1}

MODELO_SOTA = 'BAAI/bge-m3'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256

def despliegue_masivo_colab():
    print("🚀 Iniciando Fase 4: Despliegue Masivo en Colab")
    
    import os
    if not os.path.exists(PATH_INPUT_MASIVO):
        print(f"❌ No se encontró el CSV en: {PATH_INPUT_MASIVO}")
        return
    if not os.path.exists(PATH_MODELS_IN):
        print(f"❌ No se encontró el Cerebro (.pkl) en: {PATH_MODELS_IN}")
        print("💡 Tip: Sube el archivo multioutput_xgb_models_balanced.pkl a tu carpeta Datos_Corridos")
        return
        
    print("📂 Cargando los 34,000 versos desde Google Drive...")
    df_masivo = pd.read_csv(PATH_INPUT_MASIVO)
    df_masivo = df_masivo.dropna(subset=['verso_texto'])
    
    print(f"⚡ Iniciando Vectorización en [{DEVICE.upper()}]...")
    encoder = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    textos = df_masivo['verso_texto'].tolist()
    embeddings = encoder.encode(textos, batch_size=BATCH_SIZE, show_progress_bar=True, device=DEVICE, convert_to_numpy=True)
    
    print("🧠 Despertando al Cerebro Artificial (XGBoost)...")
    with open(PATH_MODELS_IN, 'rb') as f:
        modelos = pickle.load(f)
        
    for dim in DICCIONARIO_PENTADIMENSIONAL.keys():
        print(f"   ► Evaluando dimensión: {dim}...")
        xgb_model = modelos[dim]
        predicciones = xgb_model.predict(embeddings)
        df_masivo[f'intensidad_{dim}'] = predicciones
        
    print("💾 Guardando el Corpus Masivo Etiquetado en tu Drive...")
    df_masivo.to_csv(PATH_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"🎉 ¡Éxito! El archivo final está a salvo en: {PATH_OUTPUT_CSV}")

if __name__ == "__main__":
    despliegue_masivo_colab()giu