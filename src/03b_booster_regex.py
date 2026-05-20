import pandas as pd
import numpy as np
import sys
import os
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# ==========================================
# 1. CONEXIÓN UNIVERSAL AL PIPELINE (LOCAL/COLAB)
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
    
    # Blindaje de directorios
    DIR_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    DIR_VALIDATION.mkdir(parents=True, exist_ok=True)
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

# ==========================================
# 2. HIPERPARÁMETROS DEL MOTOR DE VANGUARDIA
# ==========================================
MODELO_SOTA = 'BAAI/bge-m3'
N_VERSOS_OBJETIVO = 100
MAX_VERSOS_POR_CANCION = 3
UMBRAL_INTENSIDAD = 2  # Solo tomamos versos calificados con 2 o 3 en el 03b

# Detectar dispositivo (GPU si está disponible, si no CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Dispositivo de cómputo detectado: {DEVICE.upper()}")

# ==========================================
# 3. FUNCIONES DEL MOTOR
# ==========================================

def extraer_ground_truth():
    """Lee los archivos 03b validados y extrae los versos de alta intensidad como anclas."""
    print("🔍 [Paso 1] Extrayendo 'Ground Truth' de validaciones Regex (03b)...")
    
    anclas_por_dimension = {}
    versos_cuarentena = set() # Para no repetir versos en el vectorial
    
    for dimension in DICCIONARIO_PENTADIMENSIONAL.keys():
        archivo_regex = DIR_VALIDATION / f"5_booster_{dimension}.csv"
        
        if not archivo_regex.exists():
            print(f"   ⚠️ No se encontró {archivo_regex.name}. Se usarán las palabras del diccionario para {dimension}.")
            continue
            
        try:
            df = pd.read_csv(archivo_regex)
            col_val = f"val_{dimension}"
            
            if col_val in df.columns and 'verso_texto' in df.columns:
                # Filtrar solo los de alta intensidad (ej. >= 2)
                # Convertimos a numérico, forzando errores a NaN, luego filtramos
                df[col_val] = pd.to_numeric(df[col_val], errors='coerce')
                df_alta_calidad = df[df[col_val] >= UMBRAL_INTENSIDAD]
                
                versos_ancla = df_alta_calidad['verso_texto'].dropna().tolist()
                
                if versos_ancla:
                    anclas_por_dimension[dimension] = versos_ancla
                    print(f"   ✅ {dimension}: Extraídos {len(versos_ancla)} versos ancla validados.")
                else:
                    print(f"   ⚠️ {dimension}: Ningún verso alcanzó intensidad {UMBRAL_INTENSIDAD}. Usando diccionario base.")
                
                # Todos los versos del archivo regex van a cuarentena (independientemente de su calificación)
                versos_cuarentena.update(df['verso_texto'].tolist())
                
        except Exception as e:
            print(f"   ❌ Error leyendo {archivo_regex.name}: {e}")
            
    return anclas_por_dimension, versos_cuarentena

def cargar_piscina(versos_cuarentena):
    """Carga la piscina y excluye los versos que ya pasaron por el booster regex."""
    if not PATH_PISCINA.exists():
        print(f"❌ No se encontró la piscina en: {PATH_PISCINA}")
        sys.exit(1)
        
    df_piscina = pd.read_csv(PATH_PISCINA)
    versos_iniciales = len(df_piscina)
    
    df_filtrada = df_piscina[~df_piscina['verso_texto'].isin(versos_cuarentena)].reset_index(drop=True)
    print(f"🌊 Piscina activa: {len(df_filtrada)} versos (Se excluyeron {versos_iniciales - len(df_filtrada)} ya usados en Regex).")
    
    return df_filtrada

def generar_embeddings_piscina(model, textos):
    """Genera embeddings masivos con GPU si es posible y los guarda en caché."""
    if PATH_CACHE.exists():
        print(f"⚡ [Paso 2] Recuperando embeddings de piscina desde caché ({PATH_CACHE.name})...")
        embeddings = np.load(PATH_CACHE).astype('float32')
        return embeddings
        
    print(f"🧠 [Paso 2] Calculando tensores de la piscina (Batching en {DEVICE.upper()})...")
    # Batch size alto para GPU, menor para CPU
    batch_size = 128 if DEVICE == 'cuda' else 32
    
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=batch_size, device=DEVICE, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    
    # Normalización L2 requerida para usar Inner Product (Similitud Coseno) en Faiss
    faiss.normalize_L2(embeddings)
    
    np.save(PATH_CACHE, embeddings)
    print(f"💾 Caché de embeddings guardada con éxito.")
    
    return embeddings

def calcular_centroide(model, textos_ancla):
    """Calcula el 'Centro de Gravedad' promediando los vectores de los versos ancla."""
    embeddings_ancla = model.encode(textos_ancla, batch_size=32, device=DEVICE, convert_to_numpy=True)
    # Promedio por columnas (axis=0) para obtener un solo vector representativo
    centroide = np.mean(embeddings_ancla, axis=0, keepdims=True).astype('float32')
    faiss.normalize_L2(centroide)
    return centroide

def aplicar_diversidad(df_candidatos):
    """Asegura que no dominemos la muestra con una sola canción."""
    seleccionados = []
    conteos_cancion = {}
    
    for _, row in df_candidatos.iterrows():
        track_id = f"{row['artist']}_{row['song']}"
        conteos_cancion[track_id] = conteos_cancion.get(track_id, 0) + 1
        
        if conteos_cancion[track_id] <= MAX_VERSOS_POR_CANCION:
            seleccionados.append(row)
            if len(seleccionados) == N_VERSOS_OBJETIVO:
                break
                
    return pd.DataFrame(seleccionados)

# ==========================================
# 4. FLUJO PRINCIPAL
# ==========================================
def main():
    print("🚀 INICIANDO MOTOR VECTORIAL DE CENTROIDES...")
    
    # 1. Extraer Ground Truth
    anclas_por_dimension, versos_cuarentena = extraer_ground_truth()
    
    # 2. Preparar Piscina
    df_piscina = cargar_piscina(versos_cuarentena)
    if df_piscina.empty:
        print("⚠️ La piscina está vacía tras aplicar la cuarentena.")
        return
        
    textos_piscina = df_piscina['verso_texto'].tolist()
    
    # 3. Cargar Modelo y Indexar Piscina con Faiss
    print(f"\n📦 Cargando modelo SOTA: {MODELO_SOTA}...")
    model = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    
    embeddings_piscina = generar_embeddings_piscina(model, textos_piscina)
    
    dimension_vector = embeddings_piscina.shape[1]
    
    # Intentar usar Faiss en GPU si está disponible
    res = None
    if DEVICE == 'cuda':
        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dimension_vector)
            print("🚀 Índice Faiss configurado en GPU.")
        except Exception:
            print("⚠️ Falló Faiss-GPU, usando CPU (IndexFlatIP).")
            index = faiss.IndexFlatIP(dimension_vector)
    else:
        index = faiss.IndexFlatIP(dimension_vector)
        
    index.add(embeddings_piscina)
    print("📊 Piscina indexada cuánticamente.")

    # 4. Minería por Dimensión
    for dimension, palabras_diccionario in DICCIONARIO_PENTADIMENSIONAL.items():
        archivo_salida = DIR_VALIDATION / f"5_booster_{dimension}_vectorial.csv"
        
        if archivo_salida.exists():
            print(f"\n⏭️ Booster Vectorial para {dimension} ya existe. Saltando...")
            continue
            
        print(f"\n🎯 [Paso 3] Minando dimensión: {dimension}...")
        
        # Determinar qué usar como ancla (versos validados o palabras clave de respaldo)
        if dimension in anclas_por_dimension:
            anclas = anclas_por_dimension[dimension]
            print(f"   🧭 Calculando Centroide Semántico basado en {len(anclas)} versos validados (Ground Truth).")
            vector_busqueda = calcular_centroide(model, anclas)
        else:
            print(f"   🧭 Sin validaciones. Calculando Centroide basado en el diccionario rígido.")
            vector_busqueda = calcular_centroide(model, palabras_diccionario)
            
        # 5. Búsqueda KNN
        top_k_busqueda = N_VERSOS_OBJETIVO * 3 # Buscamos más por si el filtro de diversidad elimina muchos
        if top_k_busqueda > len(textos_piscina):
            top_k_busqueda = len(textos_piscina)
            
        distancias, indices = index.search(vector_busqueda, top_k_busqueda)
        
        # Extraer candidatos y scores
        indices_crudos = indices[0]
        scores = distancias[0]
        
        df_candidatos = df_piscina.iloc[indices_crudos].copy()
        df_candidatos['score_similitud'] = scores
        
        # 6. Aplicar Diversidad (Max versos por canción)
        df_final = aplicar_diversidad(df_candidatos)
        
        # Preparar CSV final
        columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
        for col in columnas_val:
            df_final[col] = ""
            
        columnas_finales = ['artist', 'song', 'year', 'verso_texto', 'score_similitud'] + columnas_val
        df_final = df_final.reindex(columns=columnas_finales)
        
        df_final.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
        print(f"   ✅ Extraídos {len(df_final)} versos latentes (Riqueza semántica capturada).")
        
    print("\n🎉 Proceso Vectorial Finalizado.")

if __name__ == "__main__":
    main()