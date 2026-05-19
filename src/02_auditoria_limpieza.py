import pandas as pd
import re
import unicodedata
import sys
from pathlib import Path
try:
    # Importamos las rutas y diccionarios centralizados
    from config import (
        PATH_CORPUS_AGRUPADO, # <--- ESTA ES LA PIEZA QUE FALTABA
        PATH_CORPUS_MARCADO, 
        PATH_AUDITORIA_CONTEXTOS,
        SET_NARCO_FUERTE,
        UMBRAL_MINIMO_PALABRAS,
        VENTANA_CONTEXTO,
        TOP_CANCIONES_POR_DECADA
    )
    print("✅ Configuración cargada: Procesando el corpus agrupado.")
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit()

# --- BLOQUE DE CONEXIÓN CON CONFIG ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    # Importamos las rutas y diccionarios centralizados
    from config import (
        PATH_CORPUS_CRUDO, 
        PATH_CORPUS_MARCADO, 
        PATH_AUDITORIA_CONTEXTOS,
        SET_NARCO_FUERTE,
        UMBRAL_MINIMO_PALABRAS,
        VENTANA_CONTEXTO,
        TOP_CANCIONES_POR_DECADA
    )
    print("✅ Configuración cargada: Procesando el corpus completo de 11,959 canciones.")
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit()

# ==========================================
# 2. FUNCIONES DE APOYO
# ==========================================
def generar_huella_digital(artista, cancion):
    """Evita duplicados procesando nombres de forma limpia."""
    texto = f"{str(artista)}{str(cancion)}".lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]', '', texto)

def contar_palabras_narco(texto):
    """Cuenta cuántas palabras del diccionario aparecen en la letra."""
    if not isinstance(texto, str): return 0
    palabras = set(re.findall(r'\b[a-záéíóúñü0-9-]+\b', texto.lower()))
    return len(palabras.intersection(SET_NARCO_FUERTE))

def extraer_contextos(texto, cancion, artista):
    """Extrae la frase donde se encontró una palabra clave."""
    hallazgos = []
    if not isinstance(texto, str): return hallazgos
    palabras = re.findall(r'\b[a-záéíóúñü0-9-]+\b', texto.lower())
    for i, palabra in enumerate(palabras):
        if palabra in SET_NARCO_FUERTE:
            inicio, fin = max(0, i - VENTANA_CONTEXTO), min(len(palabras), i + VENTANA_CONTEXTO + 1)
            hallazgos.append({
                'Artista': artista, 'Cancion': cancion, 'Palabra': palabra,
                'Contexto': f"... {' '.join(palabras[inicio:fin])} ..."
            })
    return hallazgos

# ==========================================
# 3. EJECUCIÓN DEL PIPELINE
# ==========================================
def ejecutar_marcado():
    # 1. Cargar el corpus completo (PATH_CORPUS_CRUDO)
# Ahora el 02 lee las canciones que sobrevivieron al K-Means del 01
    df = pd.read_csv(PATH_CORPUS_AGRUPADO).dropna(subset=['lyrics'])
    
    # 2. Limpieza de duplicados
    df['huella'] = df.apply(lambda x: generar_huella_digital(x['artist'], x['song']), axis=1)
    df = df.drop_duplicates(subset=['huella']).drop(columns=['huella'])
    
    # 3. Aplicar Diccionario
    print("🔍 Buscando temáticas de crimen y violencia...")
    df['conteo_narco'] = df['lyrics'].apply(contar_palabras_narco)
    df['es_narco_fuerte'] = df['conteo_narco'] >= UMBRAL_MINIMO_PALABRAS
    
    # 4. Extraer contextos para auditoría
    datos_contexto = []
    for _, fila in df[df['es_narco_fuerte']].iterrows():
        datos_contexto.extend(extraer_contextos(fila['lyrics'], fila['song'], fila['artist']))
    
    # 5. Organizar por décadas
    df['decada'] = (pd.to_numeric(df['year'], errors='coerce') // 10) * 10
    df_final = df[df['decada'] > 0].sort_values(by=['decada', 'conteo_narco'], ascending=[True, False])
    
    # Guardar resultados
    df_final.to_csv(PATH_CORPUS_MARCADO, index=False, encoding='utf-8-sig')
    pd.DataFrame(datos_contexto).to_csv(PATH_AUDITORIA_CONTEXTOS, index=False, encoding='utf-8-sig')
    
    print(f"✅ Proceso completado. Canciones detectadas con temática narco: {df['es_narco_fuerte'].sum()}")
    print(f"💾 Archivos guardados en 02_intermediate.")

if __name__ == "__main__":
    ejecutar_marcado()