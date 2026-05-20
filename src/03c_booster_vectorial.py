import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# --- CONEXIÓN UNIVERSAL (LOCAL Y GOOGLE COLAB) ---
# Detecta dinámicamente si estás en VS Code o en Colab
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    # Fallback si se ejecuta directamente desde una celda en Colab
    dir_src = Path(os.getcwd()) / "src" 

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_INTERMEDIATE, DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    PATH_PISCINA = DIR_INTERMEDIATE / "03_piscina_versos_restantes.csv"
    PATH_CACHE = DIR_INTERMEDIATE / "cache_embeddings.npy"
    
    # ✨ BLINDAJE DE CARPETAS: Crea los directorios si Git no los trajo
    DIR_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    DIR_VALIDATION.mkdir(parents=True, exist_ok=True)
    
except ImportError as e:
    print(f"❌ Error al cargar config.py. Asegúrate de ejecutar desde la raíz del proyecto. Detalle: {e}")
    sys.exit()

# --- HIPERPARÁMETROS DEL MOTOR ---
MODELO_SOTA = 'BAAI/bge-m3'
UMBRAL_SIMILITUD = 0.55
MAX_VERSOS_POR_CANCION = 3
N_VERSOS_OBJETIVO = 100

def cargar_piscina_con_cuarentena():
    print("🔍 [Paso 1] Aplicando Filtro de Cuarentena Cruzada...")
    
    # Prevenir error si la piscina aún no existe
    if not PATH_PISCINA.exists():
        print(f"⚠️ No se encontró la piscina en: {PATH_PISCINA}")
        sys.exit()
        
    df_piscina = pd.read_csv(PATH_PISCINA)
    versos_iniciales = len(df_piscina)
    
    versos_cuarentena = set()
    for archivo in DIR_VALIDATION.glob("5_booster_*.csv"):
        try:
            df_existente = pd.read_csv(archivo)
            if 'verso_texto' in df_existente.columns:
                versos_cuarentena.update(df_existente['verso_texto'].tolist())
        except Exception:
            pass
            
    df_filtrada = df_piscina[~df_piscina['verso_texto'].isin(versos_cuarentena)].reset_index(drop=True)
    print(f"   🛡️ Versos protegidos en cuarentena: {len(versos_cuarentena)}")
    print(f"   🌊 Piscina activa: {len(df_filtrada)} de {versos_iniciales} versos.")
    
    return df_filtrada

def obtener_embeddings(model, textos, path_cache=None):
    if path_cache and os.path.exists(path_cache):
        print(f"⚡ [Paso 2] Recuperando embeddings de la caché local...")
        embeddings = np.load(path_cache)
        # Asegurar float32 explícito para compatibilidad con Faiss en Windows/Linux
        return embeddings.astype('float32')
        
    print(f"🧠 [Paso 2] Calculando vectores en lotes (Bleeding Edge)...")
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    
    if path_cache:
        np.save(path_cache, embeddings)
        print(f"💾 Caché guardada en {path_cache.name}.")
    return embeddings

def aplicar_diversidad_y_limite(df_candidatos):
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

def generar_boosters_vectoriales():
    print("🚀 Inicializando Motor Vectorial...")
    df_piscina = cargar_piscina_con_cuarentena()
    
    if df_piscina.empty:
        print("⚠️ Piscina vacía tras cuarentena. Fin del script.")
        return

    model = SentenceTransformer(MODELO_SOTA)
    
    textos_piscina = df_piscina['verso_texto'].tolist()
    embeddings_piscina = obtener_embeddings(model, textos_piscina, PATH_CACHE)
    
    dimension_vector = embeddings_piscina.shape[1]
    index = faiss.IndexFlatIP(dimension_vector) 
    index.add(embeddings_piscina)

    for dimension, palabras in DICCIONARIO_PENTADIMENSIONAL.items():
        archivo_salida = DIR_VALIDATION / f"5_booster_{dimension}_vector.csv"
        if archivo_salida.exists():
            print(f"⏭️ Booster {dimension} ya existe. Saltando...")
            continue
            
        print(f"🎯 [Paso 3 & 4] Minando dimensión: {dimension}...")
        
        embeddings_anclas = model.encode(palabras, convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(embeddings_anclas)
        
        top_k = N_VERSOS_OBJETIVO * 2
        # Blindaje por si la piscina quedó con muy pocos versos tras la cuarentena
        if top_k > len(textos_piscina):
            top_k = len(textos_piscina)
            
        distancias, indices = index.search(embeddings_anclas, top_k)
        
        resultados_crudos = {}
        for i_ancla in range(len(palabras)):
            for j_resultado in range(top_k):
                idx_verso = indices[i_ancla][j_resultado]
                score = distancias[i_ancla][j_resultado]
                
                if idx_verso not in resultados_crudos or score > resultados_crudos[idx_verso]:
                    resultados_crudos[idx_verso] = score

        indices_validos = [idx for idx, score in resultados_crudos.items() if score >= UMBRAL_SIMILITUD]
        scores_validos = [score for score in resultados_crudos.values() if score >= UMBRAL_SIMILITUD]
        
        df_candidatos = df_piscina.iloc[indices_validos].copy()
        df_candidatos['similitud_maxima'] = scores_validos
        
        df_candidatos = df_candidatos.sort_values(by='similitud_maxima', ascending=False)
        df_final = aplicar_diversidad_y_limite(df_candidatos)
        
        if df_final.empty:
            print(f"   ⚠️ Ningún verso superó el umbral de {UMBRAL_SIMILITUD} para {dimension}.")
            continue
            
        columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
        for col in columnas_val:
            df_final[col] = ""
            
        columnas_finales = ['artist', 'song', 'year', 'verso_texto'] + columnas_val
        df_final = df_final.reindex(columns=columnas_finales)
        
        df_final.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
        print(f"   ✅ Extraídos {len(df_final)} versos. ({archivo_salida.name})")

if __name__ == "__main__":
    generar_boosters_vectoriales()