import pandas as pd
import os

# ==========================================
# RUTAS EN TU GOOGLE DRIVE
# ==========================================
CARPETA_DRIVE = "/content/drive/MyDrive/Datos_Corridos"
PATH_INPUT = f"{CARPETA_DRIVE}/04_corpus_masivo_etiquetado.csv"
PATH_OUTPUT = f"{CARPETA_DRIVE}/05_corpus_canciones_condensadas.csv"

def condensar_corpus_exacto():
    print("🚀 Iniciando Módulo 08: Condensación Topológica (Ajustado a tus columnas)")
    
    if not os.path.exists(PATH_INPUT):
        print(f"❌ No se encontró el archivo masivo en: {PATH_INPUT}")
        return
        
    print("📂 Cargando los miles de versos etiquetados...")
    df = pd.read_csv(PATH_INPUT)
    
    # 1. Identificar columnas de intensidad generadas por la IA
    cols_intensidad = [c for c in df.columns if c.startswith('intensidad_')]
    
    # 2. REGLAS METODOLÓGICAS DE AGRUPACIÓN
    diccionario_agrupacion = {}
    
    # A) Metadatos: Conservamos el año (el primer valor que aparezca, ya que es igual para toda la canción)
    if 'year' in df.columns:
        diccionario_agrupacion['year'] = 'first'
            
    # B) Volumen: Contamos cuántos versos tuvo la canción originalmente
    if 'verso_texto' in df.columns:
        diccionario_agrupacion['verso_texto'] = 'count'
        
    # C) Topología (Max Pooling): Extraemos el Clímax Máximo alcanzado en la canción
    for col in cols_intensidad:
        diccionario_agrupacion[col] = 'max'

    # 💡 MEJORA METODOLÓGICA: Agrupamos por Artista Y Canción. 
    # Si dos artistas distintos tienen una canción llamada "El Jefe", esto evitará que se mezclen.
    columnas_agrupacion = ['artist', 'song']
    print(f"📊 Agrupando matemáticamente usando las llaves: {columnas_agrupacion}...")
    
    df_canciones = df.groupby(columnas_agrupacion, as_index=False).agg(diccionario_agrupacion)
    
    # Limpiamos el nombre de la columna de conteo para mayor claridad en tu análisis
    if 'verso_texto' in df_canciones.columns:
        df_canciones = df_canciones.rename(columns={'verso_texto': 'total_versos_analizados'})
    
    # Ordenamos cronológicamente para facilitar la lectura
    if 'year' in df_canciones.columns:
        df_canciones = df_canciones.sort_values('year').reset_index(drop=True)
    
    # 4. Guardar resultado final
    df_canciones.to_csv(PATH_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"🎉 ¡Efecto de Agrupación Exitoso!")
    print(f"   ► Redujimos de {len(df)} versos a {len(df_canciones)} canciones únicas.")
    print(f"   💾 Archivo listo para la tesis en: {PATH_OUTPUT}")

if __name__ == "__main__":
    condensar_corpus_exacto()