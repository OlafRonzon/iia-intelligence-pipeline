import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import entropy
import webbrowser

def calcular_entropia_anual(df_year, dimensiones):
    # Sumar las intensidades del año para obtener una "bolsa de palabras" topológica
    totales = df_year[dimensiones].sum().values
    if totales.sum() == 0: return 0
    # Convertir a distribución de probabilidad
    probabilidades = totales / totales.sum()
    return entropy(probabilidades, base=2)

def main():
    print("🔥 [Paso 18] Calculando Entropía de Shannon con Bootstrapping...")
    
    BASE_DIR = Path(os.getcwd())
    path_df = BASE_DIR / "data" / "02_intermediate" / "16_CORPUS_DEFINITIVO_TESIS.csv"
    
    df = pd.read_csv(path_df)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    
    DIMS = ['intensidad_AR_avg', 'intensidad_MO_avg', 'intensidad_DI_avg', 'intensidad_PO_avg', 'intensidad_NA_avg', 'intensidad_GO_avg']
    
    resultados = []
    anios = sorted(df['year'].unique())
    
    # BOOTSTRAPPING: 100 iteraciones por año para sacar el intervalo de confianza
    for anio in anios:
        df_anio = df[df['year'] == anio]
        n_canciones = len(df_anio)
        if n_canciones < 3: continue # Ignorar años con ruido estadístico extremo
        
        entropias_boot = []
        for _ in range(100):
            # Muestreo aleatorio con reemplazo
            muestra = df_anio.sample(n=n_canciones, replace=True)
            ent = calcular_entropia_anual(muestra, DIMS)
            entropias_boot.append(ent)
            
        resultados.append({
            'year': anio,
            'canciones': n_canciones,
            'entropia_media': np.mean(entropias_boot),
            'lim_inferior': np.percentile(entropias_boot, 5),
            'lim_superior': np.percentile(entropias_boot, 95)
        })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(BASE_DIR / "data" / "02_intermediate" / "18_entropia_shannon.csv", index=False)

    # GRAFICAR
    fig = go.Figure()
    
    # Banda de confianza (Sombra)
    fig.add_trace(go.Scatter(
        x=df_res['year'].tolist() + df_res['year'].tolist()[::-1],
        y=df_res['lim_superior'].tolist() + df_res['lim_inferior'].tolist()[::-1],
        fill='toself', fillcolor='rgba(0, 204, 150, 0.2)', line=dict(color='rgba(255,255,255,0)'),
        showlegend=False, name='Intervalo 90%'
    ))
    
    # Línea principal
    fig.add_trace(go.Scatter(
        x=df_res['year'], y=df_res['entropia_media'],
        mode='lines+markers', line=dict(color='#00CC96', width=3),
        name='Entropía Media (Diversidad Discursiva)'
    ))

    fig.update_layout(
        title="Puntos de Ebullición: Entropía de Shannon en la Topología del Agenciamiento",
        xaxis_title="Año", yaxis_title="Nivel de Entropía (Bits)",
        template="plotly_dark", hovermode="x unified"
    )
    
    path_html = BASE_DIR / "data" / "02_intermediate" / "18_grafico_entropia.html"
    fig.write_html(str(path_html))
    print(f"🎉 Gráfico interactivo exportado. Abriendo en navegador...")
    webbrowser.open('file://' + str(path_html.absolute()))

if __name__ == "__main__":
    main()