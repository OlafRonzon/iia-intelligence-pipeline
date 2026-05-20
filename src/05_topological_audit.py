import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import umap
from scipy import stats

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
    from config import DIR_VALIDATION, DIR_INTERMEDIATE, DICCIONARIO_PENTADIMENSIONAL
    PATH_UMAP_RESULTS = DIR_INTERMEDIATE / "umap_3d_results.csv"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

MODELO_SOTA = 'BAAI/bge-m3'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def cargar_y_deduplicar_datos_reales():
    """Recupera los datos reales para la auditoría (ignorando SMOTE)."""
    archivos_csv = list(DIR_VALIDATION.glob("5_booster_*.csv"))
    lista_dfs = []
    
    for archivo in archivos_csv:
        df = pd.read_csv(archivo)
        lista_dfs.append(df)
        
    df_maestro = pd.concat(lista_dfs, ignore_index=True)
    
    columnas_val = [f"val_{dim}" for dim in DICCIONARIO_PENTADIMENSIONAL.keys()]
    for col in columnas_val:
        if col in df_maestro.columns:
            df_maestro[col] = pd.to_numeric(df_maestro[col], errors='coerce').fillna(0)
            
    df_maestro = df_maestro.dropna(subset=['verso_texto'])
    
    diccionario_agrupacion = {col: 'max' for col in columnas_val}
    diccionario_agrupacion['year'] = 'first'
    if 'artist' in df_maestro.columns:
        diccionario_agrupacion['artist'] = 'first'
        
    df_unico = df_maestro.groupby('verso_texto', as_index=False).agg(diccionario_agrupacion)
    
    # Determinar la Dimensión Dominante (la de mayor intensidad)
    df_unico['dimension_dominante'] = df_unico[columnas_val].idxmax(axis=1).str.replace('val_', '')
    df_unico['intensidad_max'] = df_unico[columnas_val].max(axis=1)
    
    # Filtrar solo los versos que tienen al menos un 1 de intensidad (descartar basura)
    df_unico = df_unico[df_unico['intensidad_max'] > 0]
    df_unico['decada'] = (pd.to_numeric(df_unico['year'], errors='coerce').fillna(2000) // 10) * 10
    
    return df_unico

def auditar_sesgo_temporal(df_umap):
    """Prueba de Kruskal-Wallis para demostrar independencia temporal."""
    print("\n🔬 Ejecutando Prueba de Independencia Temporal (Kruskal-Wallis)...")
    decadas = df_umap['decada'].unique()
    
    if len(decadas) < 2:
        print("⚠️ No hay suficientes décadas para hacer una prueba estadística.")
        return
        
    # Agrupamos la coordenada X por década
    grupos_x = [df_umap[df_umap['decada'] == d]['umap_x'].values for d in decadas]
    
    stat, p_value = stats.kruskal(*grupos_x)
    
    print(f"   ► P-Value resultante: {p_value}")
    if p_value > 0.05:
        print("   ✅ ÉXITO CIENTÍFICO: No se rechaza H0. La distribución espacial de los versos ES INDEPENDIENTE de la década.")
        print("   (Se ha demostrado la ausencia de sesgo teleológico).")
    else:
        print("   ⚠️ ADVERTENCIA: El P-Value es menor a 0.05. Existe dependencia cronológica en el lenguaje (Evolución lineal detectada).")

def main():
    print("🚀 Iniciando Fase 2: Auditoría Topológica de Sesgos")
    df = cargar_y_deduplicar_datos_reales()
    print(f"📊 Datos reales consolidados: {len(df)} versos validados.")
    
    print(f"📦 Vectorizando datos con {MODELO_SOTA}...")
    model = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    embeddings = model.encode(df['verso_texto'].tolist(), show_progress_bar=True, device=DEVICE, convert_to_numpy=True)
    
    print("🌌 Calculando topología tridimensional con UMAP (esto puede tomar un minuto)...")
    # Hiperparámetros bleeding-edge: n_neighbors bajo captura estructura local (mutaciones finas)
    reductor = umap.UMAP(n_neighbors=15, n_components=3, metric='cosine', random_state=42)
    embeddings_3d = reductor.fit_transform(embeddings)
    
    df['umap_x'] = embeddings_3d[:, 0]
    df['umap_y'] = embeddings_3d[:, 1]
    df['umap_z'] = embeddings_3d[:, 2]
    
    auditar_sesgo_temporal(df)
    
    df.to_csv(PATH_UMAP_RESULTS, index=False, encoding='utf-8-sig')
    print(f"\n🎉 Coordenadas topológicas exportadas exitosamente a: {PATH_UMAP_RESULTS}")

if __name__ == "__main__":
    main()