import pandas as pd
import numpy as np
import sys
import os
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 1. CONEXIÓN UNIVERSAL (LOCAL/COLAB) Y BLINDAJE
# ==========================================
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    dir_src = Path(os.getcwd()) / "src"

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_INTERMEDIATE, DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    PATH_PISCINA = DIR_INTERMEDIATE / "03_piscina_versos_restantes.csv"
    PATH_CACHE = DIR_INTERMEDIATE / "cache_embeddings.npy"
    
    DIR_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    DIR_VALIDATION.mkdir(parents=True, exist_ok=True)
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

# ==========================================
# 2. HIPERPARÁMETROS DEL MOTOR (MÁQUINA DESEANTE V2.0)
# ==========================================
MODELO_SOTA = 'BAAI/bge-m3'
N_VERSOS_OBJETIVO = 100
MAX_K_ATRACTORES = 5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Hardware de síntesis detectado: {DEVICE.upper()}")

# ==========================================
# 3. FUNCIONES DE CARTOGRAFÍA Y SÍNTESIS
# ==========================================

def extraer_intensidades_ground_truth():
    print("🔍 [Fase 1] Mapeando intensidades validadas (Ground Truth)...")
    anclas_por_dimension = {}
    versos_cuarentena = set()
    
    for dimension in DICCIONARIO_PENTADIMENSIONAL.keys():
        archivo_regex = DIR_VALIDATION / f"5_booster_{dimension}.csv"
        
        if not archivo_regex.exists():
            continue
            
        try:
            df = pd.read_csv(archivo_regex)
            col_val = f"val_{dimension}"
            
            if col_val in df.columns and 'verso_texto' in df.columns:
                df[col_val] = pd.to_numeric(df[col_val], errors='coerce').fillna(0)
                df_activo = df[df[col_val] > 0]
                
                textos = df_activo['verso_texto'].tolist()
                pesos = df_activo[col_val].tolist()
                
                if textos:
                    anclas_por_dimension[dimension] = (textos, pesos)
                    print(f"   ✅ {dimension}: Cartografiados {len(textos)} vectores de intensidad.")
                
                versos_cuarentena.update(df['verso_texto'].dropna().tolist())
                
        except Exception as e:
            print(f"   ❌ Error en estrato {archivo_regex.name}: {e}")
            
    return anclas_por_dimension, versos_cuarentena

def cargar_piscina(versos_cuarentena):
    if not PATH_PISCINA.exists():
        print(f"❌ Ausencia de datos en: {PATH_PISCINA}")
        sys.exit(1)
        
    df_piscina = pd.read_csv(PATH_PISCINA)
    df_filtrada = df_piscina[~df_piscina['verso_texto'].isin(versos_cuarentena)].reset_index(drop=True)
    print(f"🌊 Deriva disponible: {len(df_filtrada)} versos en estado latente.")
    return df_filtrada

def generar_embeddings_piscina(model, textos):
    if PATH_CACHE.exists():
        print(f"⚡ [Fase 2] Recuperando topología semántica desde caché...")
        return np.load(PATH_CACHE).astype('float32')
        
    print(f"🧠 [Fase 2] Sintetizando vectores masivos (Batching en {DEVICE.upper()})...")
    batch_size = 128 if DEVICE == 'cuda' else 32
    
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=batch_size, device=DEVICE, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    
    faiss.normalize_L2(embeddings)
    np.save(PATH_CACHE, embeddings)
    return embeddings

def calcular_atractores_kmeans(model, textos_ancla, pesos):
    """MOTOR 1: Multiplicidad del Deseo (K-Means)."""
    embeddings = model.encode(textos_ancla, batch_size=32, device=DEVICE, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    
    n_samples = embeddings.shape[0]
    pesos_array = np.array(pesos, dtype='float32')
    
    # Riesgo/Excepción: Muy pocos datos para K-Means
    if n_samples < 3:
        print("      [K-Means] Muestra demasiado pequeña (<3). Colapsando a centroide único ponderado.")
        centroide = np.average(embeddings, axis=0, weights=pesos_array).astype('float32')
        centroide = np.expand_dims(centroide, axis=0)
        faiss.normalize_L2(centroide)
        return centroide

    best_k = 2
    best_score = -1
    max_posible_k = min(MAX_K_ATRACTORES, n_samples - 1)

    for k in range(2, max_posible_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings, sample_weight=pesos_array)
        score = silhouette_score(embeddings, labels)
        
        if score > best_score:
            best_score = score
            best_k = k

    print(f"      [K-Means] Multiplicidad óptima: {best_k} atractores (Silueta: {best_score:.4f})")
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_final.fit(embeddings, sample_weight=pesos_array)
    
    atractores = kmeans_final.cluster_centers_.astype('float32')
    faiss.normalize_L2(atractores)
    return atractores

def filtrar_candidatos_v2(df_piscina, distancias, indices):
    """MOTORES 2 y 3: Zona Ricitos de Oro y Muestreo Estratificado."""
    
    # Aplanar las matrices (por si tenemos k atractores que devolvieron n candidatos cada uno)
    flat_indices = indices.flatten()
    flat_dist = distancias.flatten()
    
    df_cand = df_piscina.iloc[flat_indices].copy()
    df_cand['similitud_coseno'] = flat_dist
    
    # MOTOR 3: La Zona Ricitos de Oro (Umbrales de Fuga Semántica)
    print("      [Motor 3] Aplicando umbrales de fuga semántica (0.60 - 0.90)...")
    df_cand = df_cand[(df_cand['similitud_coseno'] >= 0.60) & (df_cand['similitud_coseno'] <= 0.90)]
    
    # Limpiar duplicados si varios atractores capturaron el mismo verso
    df_cand = df_cand.drop_duplicates(subset=['verso_texto']).sort_values('similitud_coseno', ascending=False)
    
    # Preparar Décadas
    if 'year' not in df_cand.columns:
        df_cand['year'] = 2000
    df_cand['year'] = pd.to_numeric(df_cand['year'], errors='coerce').fillna(2000)
    df_cand['decada'] = (df_cand['year'] // 10) * 10
    
    decadas_disponibles = sorted(df_cand['decada'].unique())
    print(f"      [Motor 2] Distribuyendo entre las décadas: {decadas_disponibles}")
    
    df_por_decada = {d: df_cand[df_cand['decada'] == d].copy() for d in decadas_disponibles}
    punteros_decada = {d: 0 for d in decadas_disponibles}
    decadas_agotadas = set()
    
    seleccionados = []
    artistas_vistos = set()
    
    # MOTOR 2: Muestreo Estratificado Round-Robin
    while len(seleccionados) < N_VERSOS_OBJETIVO and len(decadas_agotadas) < len(decadas_disponibles):
        for d in decadas_disponibles:
            if d in decadas_agotadas or len(seleccionados) >= N_VERSOS_OBJETIVO:
                continue
                
            verso_encontrado = False
            while punteros_decada[d] < len(df_por_decada[d]):
                row = df_por_decada[d].iloc[punteros_decada[d]]
                punteros_decada[d] += 1
                
                # Regla de Negocio: Máximo 1 verso por artista
                if row['artist'] not in artistas_vistos:
                    seleccionados.append(row)
                    artistas_vistos.add(row['artist'])
                    verso_encontrado = True
                    break
                    
            if not verso_encontrado:
                decadas_agotadas.add(d)
                print(f"      ⚠️ Advertencia: Década {int(d)}s agotada sin más candidatos viables para el umbral.")
                
    return pd.DataFrame(seleccionados)

# ==========================================
# 4. DESPLIEGUE DEL PIPELINE
# ==========================================
def main():
    print("🚀 ACTIVANDO MÁQUINA DE SÍNTESIS VECTORIAL V2.0...")
    
    anclas_por_dimension, versos_cuarentena = extraer_intensidades_ground_truth()
    
    df_piscina = cargar_piscina(versos_cuarentena)
    if df_piscina.empty:
        print("⚠️ Piscina latente agotada.")
        return
        
    textos_piscina = df_piscina['verso_texto'].tolist()
    
    print(f"\n📦 Cargando Sintetizador de Lenguaje: {MODELO_SOTA}...")
    model = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    
    embeddings_piscina = generar_embeddings_piscina(model, textos_piscina)
    dimension_vector = embeddings_piscina.shape[1]
    
    if DEVICE == 'cuda':
        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dimension_vector)
            print("🚀 Búsqueda topológica configurada en GPU.")
        except Exception:
            index = faiss.IndexFlatIP(dimension_vector)
    else:
        index = faiss.IndexFlatIP(dimension_vector)
        
    index.add(embeddings_piscina)

    for dimension, palabras_diccionario in DICCIONARIO_PENTADIMENSIONAL.items():
        archivo_salida = DIR_VALIDATION / f"5_booster_{dimension}_v2.csv"
        
        if archivo_salida.exists():
            print(f"\n⏭️ Estrato {dimension} ya desplegado. Saltando...")
            continue
            
        print(f"\n🎯 [Fase 3] Desplegando el eje de intensidad: {dimension}...")
        
        if dimension in anclas_por_dimension:
            anclas, pesos = anclas_por_dimension[dimension]
            vectores_busqueda = calcular_atractores_kmeans(model, anclas, pesos)
        else:
            print(f"   🧭 Ausencia de conexiones previas. Usando el germen teórico rígido.")
            pesos_base = [1.0] * len(palabras_diccionario)
            vectores_busqueda = calcular_atractores_kmeans(model, palabras_diccionario, pesos_base)
            
        # Ampliamos la búsqueda radicalmente (N_OBJETIVO * 10) porque los umbrales y exclusiones descartarán muchos
        top_k_busqueda = N_VERSOS_OBJETIVO * 10
        if top_k_busqueda > len(textos_piscina):
            top_k_busqueda = len(textos_piscina)
            
        distancias, indices = index.search(vectores_busqueda, top_k_busqueda)
        
        # Filtros de deriva
        df_final = filtrar_candidatos_v2(df_piscina, distancias, indices)
        
        columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
        for col in columnas_val:
            df_final[col] = ""
            
        columnas_finales = ['artist', 'song', 'year', 'verso_texto', 'similitud_coseno'] + columnas_val
        df_final = df_final.reindex(columns=columnas_finales)
        
        df_final.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
        print(f"   ✅ Artefacto exportado: {len(df_final)} versos ({dimension}).")
        
    print("\n🎉 Arqueología Semántica Finalizada.")

if __name__ == "__main__":
    main()