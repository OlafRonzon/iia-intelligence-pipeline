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
    PATH_HTML_DASHBOARD = DIR_INTERMEDIATE / "umap_dashboard_temporal.html"
except ImportError as e:
    print(f"❌ Error importando config: {e}")
    sys.exit(1)

def generar_dashboard_temporal():
    if not PATH_UMAP_RESULTS.exists():
        print(f"❌ No se encontró el archivo: {PATH_UMAP_RESULTS}")
        print("Debes ejecutar el módulo 05 primero.")
        sys.exit(1)
        
    print("🎨 Renderizando Dashboard 4D (Topología + Tiempo)...")
    df = pd.read_csv(PATH_UMAP_RESULTS)
    
    # 1. Limpieza y orden temporal para el motor de animación
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    df = df[df['year'] > 1900] # Filtro de seguridad
    df = df.sort_values('year')
    
    # 2. Escalar visualmente la intensidad
    # Multiplicamos el valor (1, 2, 3) para que el salto visual de tamaño sea obvio
    df['tamano_visual'] = df['intensidad_max'].apply(lambda x: x * 3)

    # 3. Formatear el Tooltip (Hover)
    df['hover_text'] = df.apply(
        lambda row: f"<b>Año:</b> {row['year']}<br>"
                    f"<b>Dimensión:</b> {row['dimension_dominante']}<br>"
                    f"<b>Intensidad:</b> {row['intensidad_max']}<br>"
                    f"<b>Artista:</b> {row.get('artist', 'Desconocido')}<br><br>"
                    f"<i>\"{row['verso_texto']}\"</i>", 
        axis=1
    )

    # Fijar los límites del "universo" para que la cámara no salte al cambiar de año
    rango_x = [df['umap_x'].min() - 1, df['umap_x'].max() + 1]
    rango_y = [df['umap_y'].min() - 1, df['umap_y'].max() + 1]
    rango_z = [df['umap_z'].min() - 1, df['umap_z'].max() + 1]

    # Diccionario de colores canónicos
    color_map = {
        'AR': '#d62728', 'MO': '#1f77b4', 'DI': '#ff7f0e', 
        'PO': '#9467bd', 'NA': '#2ca02c', 'GO': '#8c564b'
    }

    # 4. Generar el Gráfico Animado
    fig = px.scatter_3d(
        df, 
        x='umap_x', y='umap_y', z='umap_z',
        color='dimension_dominante',
        size='tamano_visual',
        animation_frame='year', # EL MOTOR DEL TIEMPO
        animation_group='verso_texto',
        color_discrete_map=color_map,
        category_orders={"dimension_dominante": list(color_map.keys())},
        hover_name='hover_text',
        hover_data={'umap_x': False, 'umap_y': False, 'umap_z': False, 'dimension_dominante': False, 'tamano_visual': False, 'year': False},
        range_x=rango_x, range_y=rango_y, range_z=rango_z
    )

    # 5. Estética y Configuración de UI
    fig.update_layout(
        title="Dashboard de Evolución: Migración Topológica por Año e Intensidad",
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False, title='')
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0, xanchor="right", yanchor="top",
            pad=dict(t=50, r=10),
            buttons=[dict(label="▶ Reproducir Evolución", method="animate", args=[None, dict(frame=dict(duration=800, redraw=True), fromcurrent=True)])]
        )]
    )
    
    fig.update_traces(hovertemplate="%{hovertext}")

    # Exportar
    fig.write_html(str(PATH_HTML_DASHBOARD))
    print(f"✅ Dashboard 4D generado. Abre este archivo en tu navegador: {PATH_HTML_DASHBOARD}")

if __name__ == "__main__":
    generar_dashboard_temporal()