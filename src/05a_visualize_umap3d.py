import pandas as pd
import plotly.express as px
import sys
import os
from pathlib import Path

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    dir_src = Path(os.getcwd()) / "src"

try:
    from config import DIR_INTERMEDIATE
    PATH_UMAP_RESULTS = DIR_INTERMEDIATE / "umap_3d_results.csv"
    PATH_HTML = DIR_INTERMEDIATE / "umap_cartografia.html"
except ImportError as e:
    print(f"❌ Error importando config: {e}")
    sys.exit(1)

def generar_cartografia_3d():
    if not PATH_UMAP_RESULTS.exists():
        print(f"❌ No se encontró el archivo de coordenadas: {PATH_UMAP_RESULTS}")
        print("Debes ejecutar el módulo 05 primero.")
        sys.exit(1)
        
    print("🎨 Renderizando la Cartografía del Deseo (UMAP 3D)...")
    df = pd.read_csv(PATH_UMAP_RESULTS)
    
    # Formatear textos para el tooltip (añadir saltos de línea si el verso es largo)
    df['hover_text'] = df.apply(
        lambda row: f"<b>Dimensión:</b> {row['dimension_dominante']}<br>"
                    f"<b>Década:</b> {int(row['decada'])}<br>"
                    f"<b>Artista:</b> {row.get('artist', 'Desconocido')}<br><br>"
                    f"<i>\"{row['verso_texto']}\"</i>", 
        axis=1
    )

    # Diccionario de colores para las dimensiones
    color_map = {
        'AR': '#d62728', # Rojo (Armas)
        'MO': '#1f77b4', # Azul (Movilidad)
        'DI': '#ff7f0e', # Naranja (Ostentación)
        'PO': '#9467bd', # Morado (Poder)
        'NA': '#2ca02c', # Verde (Narcóticos)
        'GO': '#8c564b'  # Café (Gobierno)
    }

    fig = px.scatter_3d(
        df, 
        x='umap_x', 
        y='umap_y', 
        z='umap_z',
        color='dimension_dominante',
        color_discrete_map=color_map,
        hover_name='hover_text',
        hover_data={'umap_x': False, 'umap_y': False, 'umap_z': False, 'dimension_dominante': False},
        opacity=0.8,
        size_max=8
    )

    # Estética Dark Mode (Bleeding Edge)
    fig.update_layout(
        title="Cartografía del Agenciamiento Narco: Topología UMAP 3D",
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        legend=dict(
            title="Dimensiones de Intensidad",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0
        )
    )
    
    # Quitar los puntos extra de información en el hover
    fig.update_traces(hovertemplate="%{hovertext}")

    fig.write_html(str(PATH_HTML))
    print(f"✅ Cartografía generada exitosamente. Abre este archivo en tu navegador: {PATH_HTML}")

if __name__ == "__main__":
    generar_cartografia_3d()