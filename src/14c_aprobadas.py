import os
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import webbrowser

# 1. Configuración de Rutas Locales
dir_src = Path(__file__).resolve().parent
if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_INTERMEDIATE
except ImportError:
    BASE_DIR = Path(os.getcwd())
    DIR_INTERMEDIATE = BASE_DIR / "data" / "02_intermediate"

def main():
    print("📊 Generando Radiografía del Filtro Topológico...")
    
    path_corpus = DIR_INTERMEDIATE / "14_corpus_10k_inferido_v2.csv"
    if not path_corpus.exists():
        print(f"❌ No se encontró: {path_corpus}")
        return

    # 2. Cargar Datos
    df = pd.read_csv(path_corpus, encoding='utf-8-sig')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[(df['year'] >= 1970) & (df['year'] <= 2025)].copy()
    
    # 3. Clasificación Binaria Preliminar (Útil vs Ruido)
    df['categoria'] = df['densidad_semantica_v2'].apply(lambda x: 'Pasan el Filtro (>0%)' if x > 0 else 'Ruido (0%)')
    
    # Cálculos globales
    total_canciones = len(df)
    utiles = len(df[df['categoria'] == 'Pasan el Filtro (>0%)'])
    ruido = len(df[df['categoria'] == 'Ruido (0%)'])
    
    # Cálculos por año para el gráfico de barras
    df_temporal = df.groupby(['year', 'categoria']).size().unstack(fill_value=0).reset_index()
    if 'Pasan el Filtro (>0%)' not in df_temporal.columns: df_temporal['Pasan el Filtro (>0%)'] = 0
    if 'Ruido (0%)' not in df_temporal.columns: df_temporal['Ruido (0%)'] = 0

    # 4. Crear el Dashboard (2 Filas, 2 Columnas)
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=(
            f"Tasa de Supervivencia<br>(Total: {total_canciones} canciones)", 
            "Distribución de las Densidades<br>(Solo canciones que pasaron)", 
            "Volumen Histórico: Útiles vs Ruido (1970-2025)"
        ),
        row_heights=[0.4, 0.6]
    )

    # Gráfico 1: Pastel (Proporción general)
    fig.add_trace(go.Pie(
        labels=['Ruido (0%)', 'Pasan el Filtro (>0%)'],
        values=[ruido, utiles],
        marker_colors=['#444444', '#00CC96'],
        hole=0.4
    ), row=1, col=1)

    # Gráfico 2: Histograma de Densidades (Solo las útiles)
    df_utiles = df[df['densidad_semantica_v2'] > 0]
    fig.add_trace(go.Histogram(
        x=df_utiles['densidad_semantica_v2'],
        nbinsx=40,
        marker_color='#00CC96',
        name='Densidad'
    ), row=1, col=2)

    # Gráfico 3: Barras Apiladas por Año
    fig.add_trace(go.Bar(
        x=df_temporal['year'], y=df_temporal['Ruido (0%)'],
        name='Ruido (0%)', marker_color='#444444'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=df_temporal['year'], y=df_temporal['Pasan el Filtro (>0%)'],
        name='Pasan el Filtro (>0%)', marker_color='#00CC96'
    ), row=2, col=1)

    # Formato General
    fig.update_layout(
        title_text="Auditoría del Corpus: Filtrado del Modelo V2",
        template='plotly_dark',
        barmode='stack',
        showlegend=False,
        height=800
    )
    
    # Ejes
    fig.update_xaxes(title_text="Densidad Semántica", row=1, col=2)
    fig.update_xaxes(title_text="Año", row=2, col=1)
    fig.update_yaxes(title_text="Cantidad de Canciones", row=1, col=2)
    fig.update_yaxes(title_text="Cantidad de Canciones", row=2, col=1)

    # Exportar y abrir
    path_html = DIR_INTERMEDIATE / "dashboard_filtro_v2.html"
    fig.write_html(str(path_html))
    
    print(f"🎉 Gráfico exportado. Se eliminaron {ruido} canciones irrelevantes.")
    print("Abriendo en tu navegador...")
    webbrowser.open('file://' + str(path_html.absolute()))

if __name__ == "__main__":
    main()