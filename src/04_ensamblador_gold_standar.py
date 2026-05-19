import pandas as pd
import sys
from pathlib import Path

# --- CONEXIÓN DE INFRAESTRUCTURA ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    from config import DIR_VALIDATION, PATH_GOLD_STANDARD, DICCIONARIO_PENTADIMENSIONAL
    PATH_TAREAS_EXISTENTES = DIR_VALIDATION / "5_tareas_pendientes.csv"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit()

def ensamblar_gold_standard_real():
    print("🛠️ Iniciando ensamblaje del Gold Standard Hexadimensional...")

    # 1. Esquema topológico estricto basado en tus dimensiones reales
    columnas_base = ['artist', 'song', 'year', 'verso_texto']
    columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
    esquema_estricto = columnas_base + columnas_val

    # 2. Cargar tu archivo base clasificado
    if not PATH_TAREAS_EXISTENTES.exists():
        print(f"❌ No se encontró tu archivo base: {PATH_TAREAS_EXISTENTES.name}")
        return
    
    df_control = pd.read_csv(PATH_TAREAS_EXISTENTES)
    print(f"📦 Muestra control humana cargada: {len(df_control)} versos.")

    # 3. Escanear e inyectar los 6 Boosters dinámicamente
    # Busca cualquier archivo en la carpeta de validación que contenga la palabra 'booster'
    archivos_boosters = list(DIR_VALIDATION.glob("*booster*.csv"))
    print(f"🚀 Boosters detectados para fusión: {len(archivos_boosters)}")

    dataframes = [df_control]
    for archivo in archivos_boosters:
        print(f"   -> Absorbiendo: {archivo.name}")
        df_b = pd.read_csv(archivo)
        dataframes.append(df_b)

    # 4. Fusión Estructural (Concatenación vertical)
    df_maestro = pd.concat(dataframes, ignore_index=True)

    # 5. Sanitización de Datos e Intensidades (Homologación)
    if 'year' not in df_maestro.columns:
        df_maestro['year'] = 0
    df_maestro['year'] = pd.to_numeric(df_maestro['year'], errors='coerce').fillna(0).astype(int)

    # Asegurar que todas las columnas 'val_' sean numéricas y respeten tus escalas (0, 1, 2, 3)
    for col in columnas_val:
        if col not in df_maestro.columns:
            df_maestro[col] = 0
        df_maestro[col] = pd.to_numeric(df_maestro[col], errors='coerce').fillna(0).astype(int)

    # 6. Filtrado de esquema (Tira columnas sobrantes o vestigiales como las antiguas 'ia_')
    df_maestro = df_maestro.reindex(columns=esquema_estricto)

    # 7. Deduplicación Científica (Evita que el mismo verso se repita entre el control y los boosters)
    total_antes = len(df_maestro)
    df_maestro = df_maestro.drop_duplicates(subset=['verso_texto'], keep='first').reset_index(drop=True)
    duplicados_eliminados = total_antes - len(df_maestro)

    # 8. Serialización de la matriz final
    df_maestro.to_csv(PATH_GOLD_STANDARD, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("🏆 GOLD STANDARD MAESTRO GENERADO")
    print("="*50)
    print(f"🧬 Dimensión de la matriz final: {df_maestro.shape[0]} versos x {df_maestro.shape[1]} columnas.")
    if duplicados_eliminados > 0:
        print(f"🗑️ Se purgaron {duplicados_eliminados} versos duplicados entre fuentes.")
    print(f"💾 Guardado listo para Colab en: {PATH_GOLD_STANDARD.name}")

if __name__ == "__main__":
    ensamblar_gold_standard_real()