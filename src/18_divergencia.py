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
    
    # 1. Rutas locales estrictas
    BASE_DIR = Path(os.getcwd())
    DIR_INTERMEDIATE = BASE_DIR / "data" / "02_intermediate"
    
    path_df = DIR_INTERMEDIATE / "14_corpus_10k_inferido_v2.csv"
    
    if not path_df.exists():
        print(f"❌ No encuentro el archivo base en: {path_df}")
        return
        
    print("📖 Leyendo el corpus masivo...")
    df_10k = pd.read_csv(path_df, encoding='utf-8-sig')
    
    # 2. FILTRO DINÁMICO
    UMBRAL = 0.0333
    df = df_10k[df_10k['densidad_semantica_v2'] >= UMBRAL].copy()
    
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df = df[(df['year'] >= 1970) & (df['year'] <= 2025)]
    
    DIMS = ['intensidad_AR_avg', 'intensidad_MO_avg', 'intensidad_DI_avg', 'intensidad_PO_avg', 'intensidad_NA_avg', 'intensidad_GO_avg']
    
    # Agrupar en lustros (5 años)
    df['lustro'] = (df['year'] // 5) * 5
    lustros = sorted(df['lustro'].unique())
    
    resultados = []
    
    print("🧮 Midiendo la distancia matemática entre épocas...")
    for i in range(1, len(lustros)):
        lustro_previo = lustros[i-1]
        lustro_actual = lustros[i]
        
        P = obtener_distribucion(df[df['lustro'] == lustro_previo], DIMS)
        Q = obtener_distribucion(df[df['lustro'] == lustro_actual], DIMS)
        
        # JSD al cuadrado = Divergencia real
        js_div = jensenshannon(P, Q) ** 2 
        
        resultados.append({
            'periodo': f"{lustro_previo}s -> {lustro_actual}s",
            'jsd': js_div
        })

    # 3. Guardar Datos
    df_res = pd.DataFrame(resultados)
    path_csv_out = DIR_INTERMEDIATE / "19_jsd_metamorfosis.csv"
    df_res.to_csv(path_csv_out, index=False, encoding='utf-8-sig')

    # 4. Graficar
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
    
    path_html_out = DIR_INTERMEDIATE / "19_grafico_jsd.html"
    fig.write_html(str(path_html_out))
    
    print(f"💾 Resultados guardados en: {DIR_INTERMEDIATE.name}")
    print("🎉 Abriendo gráfico interactivo en el navegador...")
    webbrowser.open('file://' + str(path_html_out.absolute()))

if __name__ == "__main__":
    main()