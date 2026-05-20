import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONEXIÓN Y CONFIGURACIÓN
# ==========================================
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    dir_src = Path(os.getcwd()) / "src"

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    DIR_OUTPUT = dir_src.parent / "visualizaciones"
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

# ==========================================
# 2. CARGA INTELIGENTE DE DATOS
# ==========================================
def cargar_datos_diagnostico():
    print("🔍 Cargando estratos para diagnóstico...")
    df_regex = pd.DataFrame()
    df_vector = pd.DataFrame()
    
    columnas_val = [f"val_{dim}" for dim in DICCIONARIO_PENTADIMENSIONAL.keys()]

    for dimension in DICCIONARIO_PENTADIMENSIONAL.keys():
        # Cargar Regex (Solo validado > 0 en su dimensión principal)
        path_regex = DIR_VALIDATION / f"5_booster_{dimension}.csv"
        if path_regex.exists():
            df = pd.read_csv(path_regex)
            col_val = f"val_{dimension}"
            if col_val in df.columns:
                df[col_val] = pd.to_numeric(df[col_val], errors='coerce').fillna(0)
                df_valido = df[df[col_val] > 0].copy()
                df_valido['Origen'] = 'Regex'
                df_valido['Dimension_Base'] = dimension
                df_regex = pd.concat([df_regex, df_valido], ignore_index=True)

        # Cargar Vectorial (Todo, usando la puntuación de la IA)
        path_vector = DIR_VALIDATION / f"5_booster_{dimension}_vectorial.csv"
        if path_vector.exists():
            df = pd.read_csv(path_vector)
            df['Origen'] = 'Vectorial'
            df['Dimension_Base'] = dimension
            
            # Unificar nombre de columna de score si varió en versiones
            if 'intensidad_latente' not in df.columns and 'score_similitud' in df.columns:
                df['intensidad_latente'] = df['score_similitud']
                
            df_vector = pd.concat([df_vector, df], ignore_index=True)

    # Limpiar y preparar Regex para circularidad (rellenar NaN con 0)
    for col in columnas_val:
        if col in df_regex.columns:
            df_regex[col] = pd.to_numeric(df_regex[col], errors='coerce').fillna(0)

    # Crear décadas
    for df in [df_regex, df_vector]:
        if not df.empty:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['Decada'] = (df['year'] // 10) * 10

    return df_regex, df_vector, columnas_val

# ==========================================
# 3. DIAGNÓSTICOS VISUALES
# ==========================================

def diag_circularidad(df_regex, columnas_val):
    """Diagnóstico 1: Circularidad Semántica (Heatmap)"""
    if df_regex.empty: return
    print("   📊 Generando Heatmap de Circularidad...")
    
    # Agrupar por verso para tener el perfil multietiqueta único
    df_unico = df_regex.groupby('verso_texto')[columnas_val].max()
    corr_matrix = df_unico.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0, 
                linewidths=.5, cbar_kws={"shrink": .8})
    plt.title("Circularidad Semántica (Correlación de tus validaciones)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "04_diag_circularidad.png", dpi=300)
    plt.close()

def diag_teleologia(df_regex, df_vector):
    """Diagnóstico 2: Teleología y Sesgo del Presente"""
    print("   📊 Generando Distribución por Décadas...")
    df_total = pd.concat([df_regex, df_vector], ignore_index=True)
    if df_total.empty or 'Decada' not in df_total.columns: return

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_total, x='Decada', hue='Origen', palette={"Regex": "#2c3e50", "Vectorial": "#e74c3c"})
    plt.title("Sesgo Teleológico: Volumen de extracción por Década", fontweight='bold')
    plt.ylabel("Cantidad de Versos")
    plt.xlabel("Década")
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "05_diag_teleologia.png", dpi=300)
    plt.close()

def diag_sobrerrepresentacion(df_regex, df_vector):
    """Diagnóstico 3: Sobrerrepresentación Artística"""
    print("   📊 Generando Análisis de Artistas...")
    df_total = pd.concat([df_regex, df_vector], ignore_index=True)
    if df_total.empty: return

    top_artistas = df_total['artist'].value_counts().head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_artistas.values, y=top_artistas.index, palette="viridis")
    plt.title("Sobrerrepresentación: Top 15 Artistas en la Muestra", fontweight='bold')
    plt.xlabel("Total de Versos Extraídos")
    plt.ylabel("Artista")
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "06_diag_sobrerrepresentacion.png", dpi=300)
    plt.close()

def diag_fuga(df_vector):
    """Diagnóstico 4: Capacidad de Fuga (Violin Plot del 03c)"""
    if df_vector.empty or 'intensidad_latente' not in df_vector.columns: return
    print("   📊 Generando Mapa de Fuga (Dispersión Vectorial)...")

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_vector, x='Dimension_Base', y='intensidad_latente', 
                   palette="magma", inner="quartile")
    plt.title("Capacidad de Fuga: Distribución de similitud en la máquina deseante", fontweight='bold')
    plt.ylabel("Intensidad Latente (Cercanía matemática al centroide)")
    plt.xlabel("Dimensión")
    
    # Anotación técnica para interpretar el gráfico
    plt.figtext(0.01, 0.01, "Nota: Violines anchos/bajos indican gran fuga topológica (zona gris). Violines altos/delgados indican circularidad sobre el diccionario.", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "07_diag_fuga.png", dpi=300)
    plt.close()

# ==========================================
# 4. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print(" INICIANDO DIAGNÓSTICO DE SESGOS Y MÁQUINAS")
    print("==================================================")
    
    df_regex, df_vector, columnas_val = cargar_datos_diagnostico()
    
    diag_circularidad(df_regex, columnas_val)
    diag_teleologia(df_regex, df_vector)
    diag_sobrerrepresentacion(df_regex, df_vector)
    diag_fuga(df_vector)
    
    print("\n✅ Diagnósticos guardados en la carpeta 'visualizaciones'.")