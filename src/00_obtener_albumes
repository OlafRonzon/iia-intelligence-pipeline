import os
import time
import random
import pandas as pd
import discogs_client
import sys
from pathlib import Path
from discogs_client.exceptions import HTTPError

# --- BLOQUE DE CONEXIÓN ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    # IMPORTANTE: Añadimos ESTILOS, PAIS y AÑOS a la importación
    from config import DIR_INTERMEDIATE, ESTILOS, PAIS, AÑOS, DISCOGS_TOKEN
    print("✅ Configuración y parámetros de búsqueda cargados.")
except ImportError as e:
    print(f"❌ Error al importar desde config.py: {e}")
    sys.exit()

# ---------- RUTAS EN INTERMEDIATE ----------
# Ajustadas según tu solicitud
ARCHIVO_SALIDA = DIR_INTERMEDIATE / "PATH_DISCOGS_LIMPIO.csv"
ARCHIVO_PARCIAL = DIR_INTERMEDIATE / "PATH_DISCOGS_PROGRESO.csv"

d = discogs_client.Client('CorpusExtractor/1.0', user_token=DISCOGS_TOKEN)

def extraer_releases_año(estilo, año):
    releases = []
    page = 1
    per_page = 100
    max_retries = 5

    while True:
        retries = 0
        success = False
        pag_releases = []

        while retries < max_retries:
            try:
                results = d.search(
                    style=estilo,
                    country=PAIS,
                    year=str(año),
                    type='release',
                    per_page=per_page,
                    page=page
                )

                # Intentamos obtener la página actual
                # discogs_client carga los datos de forma perezosa (lazy)
                pag_releases = results.page(page)
                
                if not pag_releases:
                    success = True
                    break

                print(f"      Página {page}: {len(pag_releases)} resultados")

                for release in pag_releases:
                    raw = release.data if hasattr(release, 'data') else {}
                    artists_list = raw.get('artists', [])
                    artist_name = artists_list[0].get('name', 'Unknown') if artists_list else 'Unknown'
                    
                    releases.append({
                        'style': estilo,
                        'year': raw.get('year', año),
                        'artist': artist_name,
                        'title': raw.get('title', ''),
                        'discogs_id': raw.get('id', '')
                    })

                success = True
                break 

            except HTTPError as e:
                # Si el error es 404, significa que pedimos una página que no existe
                if e.status_code == 404:
                    print(f"      ℹ️ Fin de resultados (Página {page} no existe).")
                    return releases # Salimos de la función con lo que tengamos
                
                # Manejo de límite de peticiones (429)
                if e.status_code == 429:
                    wait = (2 ** retries) * 5 + random.uniform(0, 1)
                    print(f"      ⚠️ 429 Rate Limit. Esperando {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    print(f"      ⚠️ HTTPError {e.status_code}: {e}")
                    time.sleep(10)
                retries += 1
            
            except Exception as e:
                print(f"      ⚠️ Error inesperado: {e}")
                time.sleep(5)
                retries += 1

        # Si la página tiene menos de 100 resultados, es la última
        if not pag_releases or len(pag_releases) < per_page:
            break

        page += 1
        time.sleep(3 + random.uniform(0, 1))

    return releases
# ---------- EJECUCIÓN CON GUARDADO PARCIAL ----------
# ---------- EJECUCIÓN CON GUARDADO PARCIAL (RESPETANDO ESTILOS) ----------
if __name__ == "__main__":
    todos = []
    progreso_estilo_año = set()   # guarda pares (estilo, año) ya procesados

    if os.path.exists(ARCHIVO_PARCIAL):
        df_parcial = pd.read_csv(ARCHIVO_PARCIAL)
        todos = df_parcial.to_dict('records')
        # Extraemos las combinaciones únicas de estilo y año
        for _, row in df_parcial.iterrows():
            progreso_estilo_año.add((row['style'], row['year']))
        print(f"📂 Cargado progreso: {len(df_parcial)} releases "
              f"de {len(progreso_estilo_año)} combinaciones estilo-año.")
    else:
        todos = []

    for estilo in ESTILOS:
        for año in AÑOS:
            # Saltamos SOLO si ya procesamos ese estilo+año
            if (estilo, año) in progreso_estilo_año:
                print(f"\n⏩ Saltando {estilo} – {año} (ya procesado)")
                continue

            print(f"\n🔍 Buscando: {estilo} – {año}")
            datos = extraer_releases_año(estilo, año)
            print(f"   Total encontrado: {len(datos)}")
            todos.extend(datos)

            # Guardar progreso tras cada año
            pd.DataFrame(todos).to_csv(ARCHIVO_PARCIAL, index=False, encoding='utf-8-sig')

            # Pausa entre años (5‑6 segundos)
            time.sleep(5 + random.uniform(0, 2))

    # Resultado final
    df_final = pd.DataFrame(todos).drop_duplicates(subset='discogs_id')
    df_final.to_csv(ARCHIVO_SALIDA, index=False, encoding='utf-8-sig')
    print(f"\n✅ Archivo final guardado: {ARCHIVO_SALIDA}")
    print(f"   Total releases únicos: {len(df_final)}")