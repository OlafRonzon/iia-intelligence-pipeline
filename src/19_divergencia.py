import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.spatial.distance import jensenshannon
import webbrowser

def obtener_distribucion(df_lustro, dimensiones):
    totales = df_lustro[dimensiones].sum().values
    if totales.sum() == 0: return np.ones(len(dimensiones)) / len(dimensiones)
    return totales / totales.sum()

def main():
    print("🦋 [Paso 19] Calculando Metamorfosis Cualitativa (JSD)...")
    
    BASE_DIR = Path(os.getcwd())
    path_df = BASE_DIR / "data" / "02_intermediate" / "16_CORPUS_DEFINITIVO_TESIS.csv"
    
    df = pd.read_csv(path_df)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    
    DIMS = ['intensidad_AR_avg', 'intensidad_MO_avg', 'intensidad_DI_avg', 'intensidad_PO_avg', 'intensidad_NA_avg', 'intensidad_GO_avg']
    
    # Crear Lustros (Periodos de 5 años)
    df['lustro'] = (df['year'] // 5) * 5
    lustros = sorted(df['lustro'].unique())
    
    resultados = []
    
    for i in range(1, len(lustros)):
        lustro_previo = lustros[i-1]
        lustro_actual = lustros[i]
        
        P = obtener_distribucion(df[df['lustro'] == lustro_previo], DIMS)
        Q = obtener_distribucion(df[df['lustro'] == lustro_actual], DIMS)
        
        # Calcular JSD (Elevado al cuadrado da la divergencia matemática real)
        js_div = jensenshannon(P, Q) ** 2 
        
        resultados.append({
            'periodo': f"{lustro_previo}s -> {lustro_actual}s",
            'jsd': js_div
        })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(BASE_DIR / "data" / "02_intermediate" / "19_jsd_metamorfosis.csv", index=False)

    # GRAFICAR
    fig = go.Figure(data=[
        go.Bar(
            x=df_res['periodo'], 
            y=df_res['jsd'],
            marker_color='#AB63FA',
            text=df_res['jsd'].round(3),
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Metamorfosis Lingüística: Divergencia Jensen-Shannon por Lustro",
        xaxis_title="Transición de Época", 
        yaxis_title="Divergencia (JSD)",
        template="plotly_dark"
    )
    
    path_html = BASE_DIR / "data" / "02_intermediate" / "19_grafico_jsd.html"
    fig.write_html(str(path_html))
    print(f"🎉 Gráfico exportado. Abriendo en navegador...")
    webbrowser.open('file://' + str(path_html.absolute()))

if __name__ == "__main__":
    main()