import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# --- CONEXIÓN AL PIPELINE ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    from config import DIR_INTERMEDIATE, DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    PATH_PISCINA = DIR_INTERMEDIATE / "03_piscina_versos_restantes.csv"
    PATH_CACHE = DIR_INTERMEDIATE / "cache_embeddings.npy"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit()

# --- HIPERPARÁMETROS DEL MOTOR ---
MODELO_SOTA = 'BAAI/bge-m3' # Bleeding edge: Multilingüe, ligero y preciso
UMBRAL_SIMILITUD = 0.55     # Filtro rígido (Thresholding)
MAX_VERSOS_POR_CANCION = 3  # Diversidad estratégica
N_VERSOS_OBJETIVO = 100     # Tope por dimensión

def cargar_piscina_con_cuarentena():
    """Filtra los versos que ya fueron capturados por Regex u otros boosters."""
    print("🔍 [Paso 1] Aplicando Filtro de Cuarentena Cruzada...")
    df_piscina = pd.read_csv(PATH_PISCINA)
    versos_iniciales = len(df_piscina)
    
    versos_cuarentena = set()
    for archivo in DIR_VALIDATION.glob("5_booster_*.csv"):
        try:
            df_existente = pd.read_csv(archivo)
            if 'verso_texto' in df_existente.columns:
                versos_cuarentena.update(df_existente['verso_texto'].tolist())
        except Exception as e:
            pass
            
    # Purga de versos ya usados
    df_filtrada = df_piscina[~df_piscina['verso_texto'].isin(versos_cuarentena)].reset_index(drop=True)
    print(f"   🛡️ Versos protegidos en cuarentena (ya en boosters): {len(versos_cuarentena)}")
    print(f"   🌊 Piscina activa reducida de {versos_iniciales} a {len(df_filtrada)} versos.")
    
    return df_filtrada

def obtener_embeddings(model, textos, path_cache=None):
    """Vectorización con Caché para optimizar RAM y CPU."""
    if path_cache and os.path.exists(path_cache):
        print(f"⚡ [Paso 2] Recuperando embeddings de la caché local ({path_cache.name})...")
        return np.load(path_cache)
        
    print(f"🧠 [Paso 2] Calculando vectores en lotes (Bleeding Edge)...")
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    
    # Normalizamos (L2) para que Faiss pueda usar Inner Product como Coseno
    faiss.normalize_L2(embeddings)
    
    if path_cache:
        np.save(path_cache, embeddings)
        print(f"💾 Caché guardada en disco para futuras ejecuciones.")
    return embeddings

def aplicar_diversidad_y_limite(df_candidatos):
    """Filtro anti-redundancia para no saturar con una sola canción."""
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
        print("⚠️ La piscina está vacía tras la cuarentena. Fin del script.")
        return

    # Cargar modelo (Optimizado para usar CPU eficientemente)
    model = SentenceTransformer(MODELO_SOTA)
    
    # Obtener vectores (Caché activa)
    textos_piscina = df_piscina['verso_texto'].tolist()
    embeddings_piscina = obtener_embeddings(model, textos_piscina, PATH_CACHE)
    
    # Inicializar motor de búsqueda (Faiss con Inner Product para similitud coseno)
    dimension_vector = embeddings_piscina.shape[1]
    index = faiss.IndexFlatIP(dimension_vector) 
    index.add(embeddings_piscina)

    for dimension, palabras in DICCIONARIO_PENTADIMENSIONAL.items():
        archivo_salida = DIR_VALIDATION / f"5_booster_{dimension}_vector.csv"
        if archivo_salida.exists():
            print(f"⏭️ Booster para {dimension} ya existe. Saltando...")
            continue
            
        print(f"🎯 [Paso 3 & 4] Minando dimensión: {dimension} (Anclaje Individual)...")
        
        # Vectorizar anclas individualmente y normalizar
        embeddings_anclas = model.encode(palabras, convert_to_numpy=True)
        faiss.normalize_L2(embeddings_anclas)
        
        # Búsqueda ultra rápida: Extraer top N general para cada ancla
        # Buscamos el doble (N*2) para tener margen tras aplicar los filtros
        top_k = N_VERSOS_OBJETIVO * 2
        distancias, indices = index.search(embeddings_anclas, top_k)
        
        # Max-Pooling lógico: Consolidar resultados, quedarnos con la mayor similitud por verso
        resultados_crudos = {}
        for i_ancla in range(len(palabras)):
            for j_resultado in range(top_k):
                idx_verso = indices[i_ancla][j_resultado]
                score = distancias[i_ancla][j_resultado]
                
                # Solo actualizar si el score es mejor que el registrado previamente para este verso
                if idx_verso not in resultados_crudos or score > resultados_crudos[idx_verso]:
                    resultados_crudos[idx_verso] = score

        # Convertir a DataFrame y aplicar [Paso 5] Thresholding
        indices_validos = [idx for idx, score in resultados_crudos.items() if score >= UMBRAL_SIMILITUD]
        scores_validos = [score for score in resultados_crudos.values() if score >= UMBRAL_SIMILITUD]
        
        df_candidatos = df_piscina.iloc[indices_validos].copy()
        df_candidatos['similitud_maxima'] = scores_validos
        
        # Ordenar de mayor a menor calidad
        df_candidatos = df_candidatos.sort_values(by='similitud_maxima', ascending=False)
        
        # [Paso 6] Filtro de Diversidad
        df_final = aplicar_diversidad_y_limite(df_candidatos)
        
        if df_final.empty:
            print(f"   ⚠️ Ningún verso superó el umbral de {UMBRAL_SIMILITUD} para {dimension}.")
            continue
            
        # Preparar para validación manual
        columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
        for col in columnas_val:
            df_final[col] = ""
            
        # Orden final de columnas (descartamos la similitud para limpiar el CSV)
        columnas_finales = ['artist', 'song', 'year', 'verso_texto'] + columnas_val
        df_final = df_final.reindex(columns=columnas_finales)
        
        # Guardar
        df_final.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
        print(f"   ✅ Extraídos {len(df_final)} versos puros. ({archivo_salida.name})")

if __name__ == "__main__":
    generar_boosters_vectoriales()