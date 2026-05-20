import pandas as pd
import sys
import re
from pathlib import Path

# --- CONEXIÓN DE INFRAESTRUCTURA ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    from config import DIR_INTERMEDIATE, DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    PATH_PISCINA = DIR_INTERMEDIATE / "03_piscina_versos_restantes.csv"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit()

def generar_boosters_faltantes(n_versos=100):
    print("🎣 Iniciando extracción para Boosters Faltantes...")
    
    # 1. Cargar la piscina de versos remanentes
    try:
        df_piscina = pd.read_csv(PATH_PISCINA)
    except FileNotFoundError:
        print(f"❌ No existe {PATH_PISCINA.name}. Corre el Script 03 primero.")
        return

    # 2. Definir el esquema maestro para las columnas
    columnas_val = [f'val_{clave}' for clave in DICCIONARIO_PENTADIMENSIONAL.keys()]
    columnas_finales = ['artist', 'song', 'year', 'verso_texto'] + columnas_val

    # 3. Iterar sobre tu diccionario hexadimensional
    for dimension, palabras in DICCIONARIO_PENTADIMENSIONAL.items():
        # Blindaje: Revisar si ya tienes un booster para esta dimensión (ej. "5_booster_dinero_extra.csv" para "DI")
        # Busca cualquier archivo que tenga "booster" y la clave de la dimensión (sin importar mayúsculas/minúsculas)
        boosters_existentes = [f for f in DIR_VALIDATION.glob("*booster*.csv") if dimension.lower() in f.name.lower() or dimension in f.name]
        
        if boosters_existentes:
            print(f"⏭️ Booster detectado para '{dimension}' ({boosters_existentes[0].name}). Saltando extracción para proteger tu trabajo...")
            continue
            
        print(f"\n🔍 Buscando versos límite para la dimensión: {dimension}...")
        
        # 4. Construcción de Regex y Filtrado
        # Se escapan los caracteres especiales de las palabras (por si tienes "ak-47" o parecidos)
        palabras_seguras = [re.escape(pal) for pal in palabras]
        patron_regex = r'\b(' + '|'.join(palabras_seguras) + r')\b'
        
        df_filtrado = df_piscina[df_piscina['verso_texto'].str.contains(patron_regex, flags=re.IGNORECASE, na=False)].copy()
        
        if len(df_filtrado) == 0:
            print(f"⚠️ No se encontraron versos en la piscina para los términos de {dimension}.")
            continue
            
        # 5. Muestreo Aleatorio (Random Seed 42 para tu tesis)
        n_muestra = min(n_versos, len(df_filtrado))
        df_booster = df_filtrado.sample(n=n_muestra, random_state=42).copy()
        
        # 6. Homologación del esquema (Agrega columnas vacías para que califiques)
        for col in columnas_val:
            df_booster[col] = "" 
            
        df_booster = df_booster.reindex(columns=columnas_finales)
        
        # 7. Serialización en disco
        nombre_archivo = f"5_booster_{dimension}.csv"
        ruta_salida = DIR_VALIDATION / nombre_archivo
        df_booster.to_csv(ruta_salida, index=False, encoding='utf-8-sig')
        
        print(f"✅ Nuevo booster generado: {nombre_archivo} ({n_muestra} versos listos para calificar).")

    print("\n" + "="*50)
    print("🏆 EXTRACCIÓN DE BOOSTERS COMPLETADA")
    print("="*50)
    print("Por favor, revisa tu carpeta DIR_VALIDATION, llena los CSV vacíos con tus intensidades (0, 1, 2, 3) y luego ejecuta el Script 04 (Ensamblador).")

if __name__ == "__main__":
    # Extraeremos 100 versos por cada dimensión que te falte
    generar_boosters_faltantes(n_versos=100)