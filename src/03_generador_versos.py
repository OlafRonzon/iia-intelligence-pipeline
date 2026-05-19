import pandas as pd
import sys
from pathlib import Path

# --- CONEXIÓN DE INFRAESTRUCTURA ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    from config import PATH_CORPUS_MARCADO, DIR_INTERMEDIATE, DIR_VALIDATION
    PATH_TAREAS_EXISTENTES = DIR_VALIDATION / "5_tareas_pendientes_gs.csv"
    PATH_PISCINA_VERSOS = DIR_INTERMEDIATE / "03_piscina_versos_restantes.csv"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit()

def fragmentar_y_aislar_piscina():
    print("📂 Cargando tus tareas existentes y clasificadas...")
    if not PATH_TAREAS_EXISTENTES.exists():
        print(f"❌ Error crítico: El archivo '{PATH_TAREAS_EXISTENTES.name}' debe estar en tu carpeta de validación.")
        return
        
    df_tareas_humanas = pd.read_csv(PATH_TAREAS_EXISTENTES)
    set_versos_humanos = set(df_tareas_humanas['verso_texto'].dropna().str.strip())
    print(f"🎯 Tus tareas contienen {len(df_tareas_humanas)} versos clasificados por ti.")

    print("📂 Cargando canciones positivas del Stage 02 para generar la piscina...")
    try:
        df_canciones = pd.read_csv(PATH_CORPUS_MARCADO)
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo marcado del stage 02.")
        return

    # Fragmentación de todas las canciones positivas
    versos_totales = []
    df_narco = df_canciones[df_canciones['es_narco_fuerte'] == True] if 'es_narco_fuerte' in df_canciones.columns else df_canciones
    
    for _, fila in df_narco.iterrows():
        lineas = str(fila['lyrics']).split('\n')
        for idx, linea in enumerate(lineas):
            linea_limpia = linea.strip()
            if len(linea_limpia) > 20: 
                versos_totales.append({
                    'artist': fila['artist'],
                    'song': fila['song'],
                    'year': fila.get('year', 0),
                    'verso_idx': idx, 
                    'verso_texto': linea_limpia
                })
                
    df_piscina_completa = pd.DataFrame(versos_totales).drop_duplicates(subset=['verso_texto'])
    
    # INGENIERÍA INVERSA: Quitamos de la piscina masiva los versos que tú YA tienes clasificados
    # Esto garantiza que tus tareas sean únicas y el resto quede disponible para futuros boosters vectoriales
    df_piscina_restante = df_piscina_completa[~df_piscina_completa['verso_texto'].str.strip().isin(set_versos_humanos)]
    
    # Guardar la piscina remanente limpia
    df_piscina_restante.to_csv(PATH_PISCINA_VERSOS, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("✅ STAGE 03: SIMULACIÓN DE ORIGEN COMPLETADA")
    print("="*50)
    print(f"🔒 Muestra Control fija (Tus tareas): {len(df_tareas_humanas)} versos.")
    print(f"🎣 Piscina remanente aislada para boosters: {len(df_piscina_restante)} versos.")

if __name__ == "__main__":
    fragmentar_y_aislar_piscina()