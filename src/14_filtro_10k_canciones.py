import os
import sys
import pickle
import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURACIÓN HÍBRIDA (LOCAL / COLAB)
# ==========================================
try:
    import google.colab
    IN_COLAB = True
    print("☁️ Entorno: Google Colab (Aceleración CUDA activada)")
except ImportError:
    IN_COLAB = False
    print("💻 Entorno: PC Local (Windows/Linux)")

dir_src = Path(__file__).resolve().parent
if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import PATH_CORPUS_CRUDO, DIR_INTERMEDIATE
    if IN_COLAB:
        PATH_CORPUS_CRUDO = Path("/content/drive/MyDrive/Datos_Corridos/letras_corpus_final.csv")
        DIR_INTERMEDIATE = Path("/content/drive/MyDrive/Datos_Corridos")
except ImportError:
    # Fallback si no encuentra config.py
    BASE_DIR = Path(os.getcwd())
    PATH_CORPUS_CRUDO = BASE_DIR / "data" / "01_raw" / "letras_corpus_final.csv"
    DIR_INTERMEDIATE = BASE_DIR / "data" / "02_intermediate"

DIMS = ['intensidad_AR', 'intensidad_MO', 'intensidad_DI', 'intensidad_PO', 'intensidad_NA', 'intensidad_GO']

def limpiar_texto(texto):
    if pd.isna(texto): return ""
    return unicodedata.normalize('NFKC', str(texto).encode('utf-8', 'ignore').decode('utf-8')).strip()

def fragmentar_en_estrofas(letra):
    if pd.isna(letra): return []
    letra = str(letra).replace('\r\n', '\n')
    bloques = letra.split('\n\n')
    estrofas_finales = []
    
    for bloque in bloques:
        lineas = [l.strip() for l in bloque.split('\n') if l.strip()]
        if not lineas: continue
        if len(lineas) > 8:
            for i in range(0, len(lineas), 4):
                estrofas_finales.append(" \n ".join(lineas[i:i+4]))
        else:
            estrofas_finales.append(" \n ".join(lineas))
    return estrofas_finales

def main():
    print("🚀 [Script 14] Iniciando Inferencia Masiva V2 (10,000 canciones)...")
    
    path_modelo = DIR_INTERMEDIATE / "multioutput_xgb_models_v2.pkl"
    if not path_modelo.exists():
        print(f"❌ No se encontró el Modelo V2 entrenado en: {path_modelo}")
        print("Por favor, ejecuta primero el Script 13.")
        return
        
    with open(path_modelo, 'rb') as f:
        modelos_v2 = pickle.load(f)
    print("⚙️ Modelo V2 cargado correctamente con sus decodificadores topológicos.")

    # 2. Cargar y procesar Corpus de 10,000 canciones
    print(f"📖 Cargando corpus crudo desde: {PATH_CORPUS_CRUDO}")
    df_corpus = pd.read_csv(PATH_CORPUS_CRUDO, encoding='utf-8-sig')
    print(f"✅ Total canciones cargadas: {len(df_corpus)}")
    
    # 3. Explotar canciones a nivel Estrofa manteniendo trazabilidad
    print("✂️ Fragmentando canciones en unidades estróficas...")
    filas_estrofas = []
    for idx, row in df_corpus.iterrows():
        letra_limpia = limpiar_texto(row.get('lyrics', ''))
        estrofas = fragmentar_en_estrofas(letra_limpia)
        
        for i, estrofa in enumerate(estrofas):
            filas_estrofas.append({
                'song_id': idx,
                'artist': row.get('artist', ''),
                'song': row.get('song', ''),
                'year': row.get('year', ''),
                'estrofa_no': i,
                'texto_entrenamiento': estrofa
            })
            
    df_estrofas = pd.DataFrame(filas_estrofas)
    print(f"📊 Total de estrofas proyectadas para análisis: {len(df_estrofas)}")
    
    # 4. Vectorización por lotes con BGE-M3 (Aprovecha la T4 de Colab)
    print("🧠 Generando embeddings densos con BGE-M3 (Procesamiento masivo)...")
    modelo_emb = SentenceTransformer('BAAI/bge-m3')
    
    # Procesar en lotes grandes optimizados para GPU
    X = modelo_emb.encode(
        df_estrofas['texto_entrenamiento'].astype(str).tolist(), 
        batch_size=64 if IN_COLAB else 16,
        show_progress_bar=True
    )
    
    # 5. Inferencia con traducción de etiquetas
    print("🔮 Ejecutando inferencia en las 6 dimensiones topológicas...")
    for dim in DIMS:
        print(f"   -> Clasificando dimensión: {dim}")
        clf = modelos_v2[dim]['clasificador']
        le = modelos_v2[dim]['codificador']
        
        # Predecir clases mapeadas matemáticamente
        preds_encoded = clf.predict(X)
        # Decodificar a escala hermenéutica original (0, 1, 2, 3)
        df_estrofas[dim] = le.inverse_transform(preds_encoded)
        
    # Guardar dataset completo a nivel estrofa
    path_estrofas_out = DIR_INTERMEDIATE / "14_estrofas_predichas_v2.csv"
    df_estrofas.to_csv(path_estrofas_out, index=False, encoding='utf-8-sig')
    print(f"💾 Dataset de estrofas guardado en: {path_estrofas_out}")
    
    # 6. Agregación a Nivel Canción (Cálculo de nueva Densidad Semántica V2)
    print("📊 Agregando resultados a nivel canción...")
    
    # Una estrofa es "activa" si al menos una dimensión de intensidad es mayor a 0
    df_estrofas['is_active'] = (df_estrofas[DIMS].sum(axis=1) > 0).astype(int)
    
    # Agrupar por canción
    agregado = df_estrofas.groupby('song_id').agg(
        total_estrofas=('is_active', 'count'),
        estrofas_activas=('is_active', 'sum'),
        **{f'{dim}_max': (dim, 'max') for dim in DIMS},
        **{f'{dim}_avg': (dim, 'mean') for dim in DIMS}
    ).reset_index()
    
    # Calcular la Densidad Semántica V2
    agregado['densidad_semantica_v2'] = agregado['estrofas_activas'] / agregado['total_estrofas']
    
    # Unir de vuelta con los metadatos originales de las canciones
    df_corpus_final = df_corpus.drop(columns=['lyrics'], errors='ignore').copy()
    df_corpus_final = df_corpus_final.merge(agregado, left_index=True, right_on='song_id', how='left')
    
    # Guardar el corpus inferido final
    path_corpus_out = DIR_INTERMEDIATE / "14_corpus_10k_inferido_v2.csv"
    df_corpus_final.to_csv(path_corpus_out, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"🎉 ¡INFERENCIA MASIVA V2 FINALIZADA CON ÉXITO!")
    print(f"1. Archivo de estrofas completas: {path_estrofas_out.name}")
    print(f"2. Archivo del corpus de canciones analizado: {path_corpus_out.name}")
    print("="*50)

if __name__ == '__main__':
    main()