import asyncio
import aiohttp
import pandas as pd
import os, re, random
import sys
from urllib.parse import quote
from langdetect import detect
from pathlib import Path

# --- BLOQUE DE CONEXIÓN E INFRAESTRUCTURA ---
directorio_actual = Path(__file__).resolve().parent
if str(directorio_actual) not in sys.path:
    sys.path.append(str(directorio_actual))

try:
    from config import DISCOGS_TOKEN, PATH_CORPUS_CRUDO
except ImportError:
    # Fallback para Colab si no detecta el archivo config.py
    DISCOGS_TOKEN = os.getenv("DISCOGS_TOKEN", "TU_TOKEN_AQUI")
    PATH_CORPUS_CRUDO = "/content/drive/MyDrive/letras_corpus_final.csv"

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
ARCHIVO_SALIDA = PATH_CORPUS_CRUDO

ESTILOS_OBJETIVO = ['Corrido', 'Norteño', 'Trap', 'Hip Hop', 'Rap']
MIN_CARACTERES = 150
TAMANO_BUFFER = 50

LIMITE_CONCURRENCIA = asyncio.Semaphore(5)

# ==========================================
# 2. MOTORES OPTIMIZADOS CON FILTRO DE IDIOMA
# ==========================================

def guardar_en_disco(buffer_datos):
    if not buffer_datos:
        return
    df = pd.DataFrame(buffer_datos)
    
    # Crea las carpetas necesarias en VS Code si no existen (ej. data/01_raw/)
    os.makedirs(os.path.dirname(ARCHIVO_SALIDA), exist_ok=True)
    
    df.to_csv(ARCHIVO_SALIDA, mode='a', index=False, header=not os.path.exists(ARCHIVO_SALIDA), encoding='utf-8-sig')
    buffer_datos.clear()
    print(f"\n💾 [SISTEMA] ¡ÉXITO! Se ha guardado un bloque de canciones en:\n{ARCHIVO_SALIDA}\n")

async def buscar_en_lrclib(session, artista, cancion):
    url = f"https://lrclib.net/api/get?artist_name={quote(artista)}&track_name={quote(cancion)}"
    try:
        async with session.get(url, timeout=8) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get('plainLyrics') or data.get('syncedLyrics')
    except:
        pass
    return None

async def procesar_cancion(session, artista_limpio, album_nombre, año_release, t, buffer_datos):
    async with LIMITE_CONCURRENCIA:
        letra = await buscar_en_lrclib(session, artista_limpio, t)

        if letra and len(letra) >= MIN_CARACTERES:
            # --- INICIO DEL FILTRO DE IDIOMA ---
            try:
                idioma = detect(letra)
            except:
                idioma = 'desconocido'

            if idioma != 'es':
                print(f"  🚫 DESCARTADA (Idioma '{idioma}'): '{t}'")
                return 
            # --- FIN DEL FILTRO DE IDIOMA ---

            buffer_datos.append({'artist': artista_limpio, 'album': album_nombre, 'year': año_release, 'song': t, 'lyrics': letra})
            print(f"  ✅ ¡GUARDADA!: '{t}'")
        elif letra:
            print(f"  ❌ LETRA MUY CORTA: '{t}'")
        else:
            print(f"  ❌ NO ENCONTRADA: '{t}'")

async def obtener_albumes_discogs(session, estilo, año):
    url = "https://api.discogs.com/database/search"
    params = {
        'style': estilo,
        'year': año,
        'type': 'release',
        'country': 'Mexico',
        'token': DISCOGS_TOKEN,
        'per_page': 100
    }
    try:
        async with session.get(url, params=params, timeout=15) as resp:
            if resp.status == 200:
                data = await resp.json()
                res = data.get('results', [])
                print(f"\n{'='*50}\n-> [Discogs] Encontrados: {len(res)} álbumes de {estilo} ({año}) en MÉXICO\n{'='*50}")
                return res
            elif resp.status == 429:
                print("\n-> ⏳ Límite de Discogs alcanzado, esperando 30 segundos...")
                await asyncio.sleep(30)
    except Exception as e:
        print(f"\n-> ❌ Error de conexión: {str(e)[:50]}")
    return []

async def procesar_release(session, release, buffer_datos):
    try:
        partes = release['title'].split(' - ')
        artista_limpio = re.sub(r'\*|\(\d+\)', '', partes[0]).strip()
        album_nombre = partes[1] if len(partes) > 1 else "Single/Album"
        año_release = release.get('year', 'Desconocido')

        url_detalles = f"https://api.discogs.com/releases/{release['id']}"
        headers = {'Authorization': f'Discogs token={DISCOGS_TOKEN}'}

        async with session.get(url_detalles, headers=headers, timeout=12) as resp:
            if resp.status == 200:
                data = await resp.json()
                tracks = [t['title'] for t in data.get('tracklist', []) if t.get('title')]
                print(f"\n🎵 Analizando Álbum: '{album_nombre}' de {artista_limpio} ({len(tracks)} canciones)")

                if tracks:
                    tareas = [procesar_cancion(session, artista_limpio, album_nombre, año_release, t, buffer_datos) for t in tracks]
                    await asyncio.gather(*tareas)

            elif resp.status == 429:
                print("  -> ⏳ Límite de la API alcanzado al ver álbum, esperando 30s...")
                await asyncio.sleep(30)
    except Exception as e:
        print(f"  -> ❌ Error procesando el álbum: {str(e)[:50]}")

# ==========================================
# 3. MAIN
# ==========================================

async def main():
    buffer_datos = []
    
    os.makedirs(os.path.dirname(ARCHIVO_SALIDA), exist_ok=True)

    if os.path.exists(ARCHIVO_SALIDA):
        try:
            df_old = pd.read_csv(ARCHIVO_SALIDA)
            print(f"♻️ Memoria: Tienes {len(df_old)} canciones ya guardadas.\n")
        except:
            pass

    async with aiohttp.ClientSession() as session:
        todos_los_años = list(range(1970, 2027))
        random.shuffle(todos_los_años)

        for año in todos_los_años:
            for estilo in ESTILOS_OBJETIVO:
                releases = await obtener_albumes_discogs(session, estilo, año)

                for rel in releases:
                    await procesar_release(session, rel, buffer_datos)
                    await asyncio.sleep(0.8)

                    if len(buffer_datos) >= TAMANO_BUFFER:
                        guardar_en_disco(buffer_datos)

        guardar_en_disco(buffer_datos)

if __name__ == "__main__":
    try:
        # Ejecución normal en terminal (VS Code)
        asyncio.run(main())
    except RuntimeError:
        # Ejecución en Google Colab (Evita el error de 'event loop is already running')
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())