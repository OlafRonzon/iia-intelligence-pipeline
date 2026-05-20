import pandas as pd
import numpy as np
import sys
import os
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

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
    
    # Asegurar que el sustrato arqueológico (carpetas) exista
    DIR_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    DIR_VALIDATION.mkdir(parents=True, exist_ok=True)
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

# ==========================================
# 2. HIPERPARÁMETROS DEL MOTOR (MÁQUINA DESEANTE)
# ==========================================
MODELO_SOTA = 'BAAI/bge-m3'
N_VERSOS_OBJETIVO = 100
MAX_VERSOS_POR_CANCION = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Hardware de síntesis detectado: {DEVICE.upper()}")

# ==========================================
# 3. FUNCIONES DE CARTOGRAFÍA Y SÍNTESIS
# ==========================================

def extraer_intensidades_ground_truth():
    """Extrae textos y PESOS (1,2,3) para capturar el agenciamiento."""
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
                
                # Rescatamos todo eje de intensidad activo (> 0)
                df_activo = df[df[col_val] > 0]
                
                textos = df_activo['verso_texto'].tolist()
                pesos = df_activo[col_val].tolist()
                
                if textos:
                    anclas_por_dimension[dimension] = (textos, pesos)
                    print(f"   ✅ {dimension}: Cartografiados {len(textos)} vectores de intensidad.")
                
                # Lo analizado se pliega (cuarentena) para no repetirse
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
    """Calcula el espacio vectorial plegado."""
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

def calcular_sintetizador_intensidad(model, textos_ancla, pesos):
    """El Centroide como Máquina Deseante: Un promedio impulsado por la gravedad de las calificaciones."""
    embeddings_ancla = model.encode(textos_ancla, batch_size=32, device=DEVICE, convert_to_numpy=True)
    
    pesos_array = np.array(pesos, dtype='float32')
    
    # La síntesis matemática del agenciamiento
    centroide = np.average(embeddings_ancla, axis=0, weights=pesos_array).astype('float32')
    
    centroide = np.expand_dims(centroide, axis=0)
    faiss.normalize_L2(centroide)
    return centroide

def aplicar_deriva_descentrada(df_candidatos):
    """Filtro anti-monopolio: Evita que el agenciamiento se estanque en una sola canción."""
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
# 4. DESPLIEGUE DEL PIPELINE
# ==========================================
def main():
    print("🚀 ACTIVANDO MÁQUINA DE SÍNTESIS VECTORIAL...")
    
    # 1. Recuperar intensidades previas
    anclas_por_dimension, versos_cuarentena = extraer_intensidades_ground_truth()
    
    # 2. Cargar material no analizado
    df_piscina = cargar_piscina(versos_cuarentena)
    if df_piscina.empty:
        print("⚠️ Piscina latente agotada.")
        return
        
    textos_piscina = df_piscina['verso_texto'].tolist()
    
    # 3. Modelado Topológico
    print(f"\n📦 Cargando Sintetizador de Lenguaje: {MODELO_SOTA}...")
    model = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    
    embeddings_piscina = generar_embeddings_piscina(model, textos_piscina)
    dimension_vector = embeddings_piscina.shape[1]
    
    # Aceleración Cuántica con Faiss
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

    # 4. Minería Arqueológica por Dimensión
    for dimension, palabras_diccionario in DICCIONARIO_PENTADIMENSIONAL.items():
        archivo_salida = DIR_VALIDATION / f"5_booster_{dimension}_vectorial.csv"
        
        if archivo_salida.exists():
            print(f"\n⏭️ Estrato {dimension} ya desplegado. Saltando...")
            continue
            
        print(f"\n🎯 [Fase 3] Desplegando el eje de intensidad: {dimension}...")
        
        # Calcular el vector que funge como atractor/sintetizador
        if dimension in anclas_por_dimension:
            anclas, pesos = anclas_por_dimension[dimension]
            print(f"   🧭 Construyendo atractor semántico basado en {len(anclas)} conexiones humanas.")
            vector_busqueda = calcular_sintetizador_intensidad(model, anclas, pesos)
        else:
            print(f"   🧭 Ausencia de conexiones previas. Usando el germen teórico rígido.")
            pesos_base = [1.0] * len(palabras_diccionario)
            vector_busqueda = calcular_sintetizador_intensidad(model, palabras_diccionario, pesos_base)
            
        # 5. Buscar en el estado plegado (KNN)
        top_k_busqueda = N_VERSOS_OBJETIVO * 3
        if top_k_busqueda > len(textos_piscina):
            top_k_busqueda = len(textos_piscina)
            
        distancias, indices = index.search(vector_busqueda, top_k_busqueda)
        
        # Extraer candidatos
        df_candidatos = df_piscina.iloc[indices[0]].copy()
        df_candidatos['intensidad_latente'] = distancias[0]
        
        # 6. Filtrar deriva
        df_final = aplicar_deriva_descentrada(df_candidatos)
        
        # Formatear el artefacto de validación
        columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
        for col in columnas_val:
            df_final[col] = ""
            
        columnas_finales = ['artist', 'song', 'year', 'verso_texto', 'intensidad_latente'] + columnas_val
        df_final = df_final.reindex(columns=columnas_finales)
        
        df_final.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
        print(f"   ✅ Artefacto exportado: {len(df_final)} versos conectados desde el estado latente.")
        
    print("\n🎉 Arqueología Semántica Finalizada.")

if __name__ == "__main__":
    main()