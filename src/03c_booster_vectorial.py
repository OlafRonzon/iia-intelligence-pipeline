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
    # Intenta resolver la ruta si se ejecuta como script
    dir_src = Path(__file__).resolve().parent
except NameError:
    # Alternativa si se ejecuta en entornos interactivos como Jupyter/Colab
    dir_src = Path(os.getcwd()) / "src"

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    # Importamos variables de configuración
    from config import DIR_INTERMEDIATE, DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    PATH_PISCINA = DIR_INTERMEDIATE / "03_piscina_versos_restantes.csv"
    PATH_CACHE = DIR_INTERMEDIATE / "cache_embeddings.npy"
    
    # Crea las carpetas si no existen (buenas prácticas para evitar errores)
    DIR_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    DIR_VALIDATION.mkdir(parents=True, exist_ok=True)
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

# ==========================================
# 2. HIPERPARÁMETROS DEL MOTOR 
# ==========================================
MODELO_SOTA = 'BAAI/bge-m3'
N_VERSOS_OBJETIVO = 100
MAX_K_ATRACTORES = 5

# Detectar automáticamente si usamos Tarjeta Gráfica (GPU) o Procesador (CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ Hardware de síntesis detectado: {DEVICE.upper()}")

# ==========================================
# 3. FUNCIONES DE CARTOGRAFÍA Y SÍNTESIS
# ==========================================

def extraer_intensidades_ground_truth():
    """
    Extrae intensidades aprovechando la validación cruzada (multi-etiqueta).
    Cualquier verso validado en cualquier archivo alimentará su dimensión correspondiente.
    """
    print("🔍 [Fase 1] Mapeando intensidades validadas cruzadas (Omnidireccional)...")
    
    anclas_por_dimension = {}
    
    # 1. Leer todos los archivos de validación disponibles y unirlos
    dfs_validados = []
    archivos_csv = list(DIR_VALIDATION.glob("5_booster_*.csv"))
    
    if not archivos_csv:
        print("   ⚠️ No se encontraron archivos de validación en la carpeta.")
        return {}, set()
        
    for archivo in archivos_csv:
        try:
            df = pd.read_csv(archivo)
            dfs_validados.append(df)
        except Exception as e:
            print(f"   ❌ Error al leer {archivo.name}: {e}")
            
    # Unimos todos los CSV en una sola tabla maestra
    df_maestro = pd.concat(dfs_validados, ignore_index=True)
    
    # Asegurarnos de que las columnas val_ existan y sean numéricas
    columnas_val = [f"val_{dim}" for dim in DICCIONARIO_PENTADIMENSIONAL.keys()]
    for col in columnas_val:
        if col in df_maestro.columns:
            df_maestro[col] = pd.to_numeric(df_maestro[col], errors='coerce').fillna(0)
        else:
            df_maestro[col] = 0 # Si la columna no existía en ningún CSV, la creamos en ceros
            
    # 2. Agrupar por verso para colapsar duplicados (tomando la intensidad máxima)
    if 'verso_texto' not in df_maestro.columns:
        print("   ❌ Error crítico: Ningún archivo tiene la columna 'verso_texto'.")
        return {}, set()
        
    df_maestro = df_maestro.dropna(subset=['verso_texto'])
    # Si un verso aparece en 2 archivos, conservamos el valor más alto que se le haya dado
    df_maestro = df_maestro.groupby('verso_texto', as_index=False)[columnas_val].max()
    
    # 3. Extraer los vectores por dimensión transversalmente
    for dimension in DICCIONARIO_PENTADIMENSIONAL.keys():
        col_val = f"val_{dimension}"
        
        # Filtramos solo los versos que tienen un peso mayor a 0 para esta dimensión específica
        df_activo = df_maestro[df_maestro[col_val] > 0]
        
        textos = df_activo['verso_texto'].tolist()
        pesos = df_activo[col_val].tolist()
        
        if textos:
            anclas_por_dimension[dimension] = (textos, pesos)
            print(f"   ✅ {dimension}: Cartografiados {len(textos)} vectores (aprovechando validación cruzada).")
            
    # La cuarentena (versos que ya no queremos en la piscina) son todos los del df_maestro
    versos_cuarentena = set(df_maestro['verso_texto'].tolist())
    
    return anclas_por_dimension, versos_cuarentena

def cargar_piscina(versos_cuarentena):
    """Carga los versos que no han sido analizados aún."""
    if not PATH_PISCINA.exists():
        print(f"❌ Ausencia de datos en: {PATH_PISCINA}")
        sys.exit(1)
        
    df_piscina = pd.read_csv(PATH_PISCINA)
    # Filtramos los versos que ya pasaron por cuarentena
    df_filtrada = df_piscina[~df_piscina['verso_texto'].isin(versos_cuarentena)].reset_index(drop=True)
    print(f"🌊 Deriva disponible: {len(df_filtrada)} versos en estado latente.")
    return df_filtrada

def generar_embeddings_piscina(model, textos):
    """
    Genera y guarda en caché los vectores de los versos.
    Incluye una validación de seguridad para evitar desfases de índices (Out of Bounds).
    """
    # 1. Comprobamos si el archivo de caché ya existe en el disco duro
    if PATH_CACHE.exists():
        print(f"⚡ [Fase 2] Recuperando topología semántica desde caché...")
        # Cargamos los datos del caché
        embeddings_cacheados = np.load(PATH_CACHE).astype('float32')
        
        # 2. REGLA DE SEGURIDAD: Validar que el tamaño coincida
        if len(embeddings_cacheados) == len(textos):
            print("   ✅ El caché coincide con los datos actuales. Usando caché seguro.")
            return embeddings_cacheados
        else:
            # Si hay un desfase, alertamos y procedemos a recalcular
            print(f"   ⚠️ Desfase de datos detectado: Caché tiene {len(embeddings_cacheados)} vectores, pero la tabla tiene {len(textos)} filas.")
            print("   🔄 Regenerando vectores para sincronizar...")
    else:
        print(f"🧠 [Fase 2] Sintetizando vectores masivos (Batching en {DEVICE.upper()})...")
        
    # 3. Cálculo de nuevos vectores (si no hay caché o si hubo desfase)
    batch_size = 128 if DEVICE == 'cuda' else 32
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=batch_size, device=DEVICE, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    
    # Normalizamos (requerido para similitud coseno) y sobrescribimos el caché viejo
    faiss.normalize_L2(embeddings)
    np.save(PATH_CACHE, embeddings)
    
    print("   ✅ Nuevo caché guardado exitosamente.")
    return embeddings

def calcular_atractores_kmeans(model, textos_ancla, pesos):
    """Usa Machine Learning (K-Means) para crear múltiples centros semánticos."""
    embeddings = model.encode(textos_ancla, batch_size=32, device=DEVICE, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    
    n_samples = embeddings.shape[0]
    pesos_array = np.array(pesos, dtype='float32')
    
    # Manejo de excepción: Si hay menos de 3 versos, K-means fallaría. Volvemos al promedio.
    if n_samples < 3:
        print("      [K-Means] Muestra demasiado pequeña (<3). Colapsando a centroide único ponderado.")
        centroide = np.average(embeddings, axis=0, weights=pesos_array).astype('float32')
        centroide = np.expand_dims(centroide, axis=0)
        faiss.normalize_L2(centroide)
        return centroide

    # Lógica para encontrar el número de clústeres (k) ideal usando la métrica "Silhouette"
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
    """Filtra los resultados basándose en umbrales matemáticos y asegura diversidad por décadas y artistas."""
    # Aplanar las matrices de FAISS
    flat_indices = indices.flatten()
    flat_dist = distancias.flatten()
    
    df_cand = df_piscina.iloc[flat_indices].copy()
    df_cand['similitud_coseno'] = flat_dist
    
    # Filtro Ricitos de Oro: Desechar lo muy obvio (>0.90) y lo que es basura (<0.60)
    print("      [Filtro] Aplicando umbrales de similitud (0.60 - 0.90)...")
    df_cand = df_cand[(df_cand['similitud_coseno'] >= 0.60) & (df_cand['similitud_coseno'] <= 0.90)]
    df_cand = df_cand.drop_duplicates(subset=['verso_texto']).sort_values('similitud_coseno', ascending=False)
    
    # Preparar el muestreo por décadas
    if 'year' not in df_cand.columns:
        df_cand['year'] = 2000
    df_cand['year'] = pd.to_numeric(df_cand['year'], errors='coerce').fillna(2000)
    df_cand['decada'] = (df_cand['year'] // 10) * 10
    
    decadas_disponibles = sorted(df_cand['decada'].unique())
    print(f"      [Filtro] Distribuyendo equitativamente entre décadas: {decadas_disponibles}")
    
    df_por_decada = {d: df_cand[df_cand['decada'] == d].copy() for d in decadas_disponibles}
    punteros_decada = {d: 0 for d in decadas_disponibles}
    decadas_agotadas = set()
    
    seleccionados = []
    artistas_vistos = set()
    
    # Bucle de reparto equitativo (Round-Robin)
    while len(seleccionados) < N_VERSOS_OBJETIVO and len(decadas_agotadas) < len(decadas_disponibles):
        for d in decadas_disponibles:
            if d in decadas_agotadas or len(seleccionados) >= N_VERSOS_OBJETIVO:
                continue
                
            verso_encontrado = False
            while punteros_decada[d] < len(df_por_decada[d]):
                row = df_por_decada[d].iloc[punteros_decada[d]]
                punteros_decada[d] += 1
                
                # Regla estricta: Solo 1 verso por artista permitido
                if row['artist'] not in artistas_vistos:
                    seleccionados.append(row)
                    artistas_vistos.add(row['artist'])
                    verso_encontrado = True
                    break
                    
            if not verso_encontrado:
                decadas_agotadas.add(d)
                # print(f"      ⚠️ Advertencia: Década {int(d)}s agotada sin más candidatos.")
                
    return pd.DataFrame(seleccionados)

# ==========================================
# 4. DESPLIEGUE DEL PIPELINE PRINCIPAL
# ==========================================
def main():
    print("🚀 INICIANDO SCRIPT V2.0...")
    
    # 1. Extraer los datos etiquetados (Aquí es donde actuará la Auditoría)
    anclas_por_dimension, versos_cuarentena = extraer_intensidades_ground_truth()
    
    # 2. Cargar los datos nuevos
    df_piscina = cargar_piscina(versos_cuarentena)
    if df_piscina.empty:
        print("⚠️ Piscina latente agotada.")
        return
        
    textos_piscina = df_piscina['verso_texto'].tolist()
    
    # 3. Preparar el modelo de Inteligencia Artificial
    print(f"\n📦 Cargando Modelo Lingüístico: {MODELO_SOTA}...")
    model = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    
    embeddings_piscina = generar_embeddings_piscina(model, textos_piscina)
    dimension_vector = embeddings_piscina.shape[1]
    
    # Configurar FAISS para búsquedas hiper-rápidas
    if DEVICE == 'cuda':
        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, dimension_vector)
            print("🚀 Búsqueda configurada en Tarjeta Gráfica (GPU).")
        except Exception:
            index = faiss.IndexFlatIP(dimension_vector)
    else:
        index = faiss.IndexFlatIP(dimension_vector)
        
    index.add(embeddings_piscina)

    # 4. Iterar por cada dimensión y buscar candidatos
    for dimension, palabras_diccionario in DICCIONARIO_PENTADIMENSIONAL.items():
        archivo_salida = DIR_VALIDATION / f"5_booster_{dimension}_v2.csv"
        
        # Evitar sobreescribir trabajo ya hecho
        if archivo_salida.exists():
            print(f"\n⏭️ La dimensión {dimension} ya fue procesada. Saltando...")
            continue
            
        print(f"\n🎯 [Fase 3] Procesando la dimensión: {dimension}...")
        
        # Si la auditoría fue exitosa, usamos K-Means. Si falló, usamos las palabras base.
        if dimension in anclas_por_dimension:
            anclas, pesos = anclas_por_dimension[dimension]
            vectores_busqueda = calcular_atractores_kmeans(model, anclas, pesos)
        else:
            print(f"   🧭 Ausencia de conexiones previas. Usando palabras base del diccionario.")
            pesos_base = [1.0] * len(palabras_diccionario)
            vectores_busqueda = calcular_atractores_kmeans(model, palabras_diccionario, pesos_base)
            
        # Extraemos 10 veces más versos porque los filtros eliminarán a muchos
        top_k_busqueda = N_VERSOS_OBJETIVO * 10
        if top_k_busqueda > len(textos_piscina):
            top_k_busqueda = len(textos_piscina)
            
        distancias, indices = index.search(vectores_busqueda, top_k_busqueda)
        
        # Aplicamos reglas de negocio (Umbrales y Artistas únicos)
        df_final = filtrar_candidatos_v2(df_piscina, distancias, indices)
        
        # Preparamos el archivo para que puedas validarlo a mano (creando las columnas vacías)
        columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
        for col in columnas_val:
            df_final[col] = ""
            
        columnas_finales = ['artist', 'song', 'year', 'verso_texto', 'similitud_coseno'] + columnas_val
        df_final = df_final.reindex(columns=columnas_finales)
        
        # Exportar CSV final
        df_final.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
        print(f"   ✅ Archivo guardado con éxito: {len(df_final)} versos ({dimension}).")
        
    print("\n🎉 Proceso Finalizado. Revisa tus archivos CSV generados.")

# Punto de entrada estándar en Python
if __name__ == "__main__":
    main()