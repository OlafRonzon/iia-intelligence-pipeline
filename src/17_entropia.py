import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import entropy
import webbrowser

def calcular_entropia_anual(df_year, dimensiones):
    totales = df_year[dimensiones].sum().values
    if totales.sum() == 0: return 0
    probabilidades = totales / totales.sum()
    return entropy(probabilidades, base=2)

def main():
    print("🔥 [Paso 18] Calculando Entropía de Shannon con Bootstrapping...")
    
    # 1. Rutas locales estrictas
    BASE_DIR = Path(os.getcwd())
    DIR_INTERMEDIATE = BASE_DIR / "data" / "02_intermediate"
    
    path_df = DIR_INTERMEDIATE / "14_corpus_10k_inferido_v2.csv"
    
    if not path_df.exists():
        print(f"❌ No encuentro el archivo base en: {path_df}")
        print("Asegúrate de ejecutar el script desde la raíz del proyecto (la carpeta que contiene 'data' y 'src').")
        return
        
    print("📖 Leyendo el corpus masivo...")
    df_10k = pd.read_csv(path_df, encoding='utf-8-sig')
    
    # 2. FILTRO DINÁMICO (Aislando la "Materia Prima Pura")
    UMBRAL = 0.0333
    df = df_10k[df_10k['densidad_semantica_v2'] >= UMBRAL].copy()
    print(f"✂️ Analizando {len(df)} canciones con agenciamiento confirmado...")
    
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df = df[(df['year'] >= 1970) & (df['year'] <= 2025)]
    
    DIMS = ['intensidad_AR_avg', 'intensidad_MO_avg', 'intensidad_DI_avg', 'intensidad_PO_avg', 'intensidad_NA_avg', 'intensidad_GO_avg']
    
    resultados = []
    anios = sorted(df['year'].unique())
    
    print("🧮 Ejecutando remuestreo matemático (Bootstrapping)...")
    for anio in anios:
        df_anio = df[df['year'] == anio]
        n_canciones = len(df_anio)
        if n_canciones < 3: continue 
        
        entropias_boot = []
        for _ in range(100):
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

    # 3. Guardar Datos
    df_res = pd.DataFrame(resultados)
    path_csv_out = DIR_INTERMEDIATE / "18_entropia_shannon.csv"
    df_res.to_csv(path_csv_out, index=False, encoding='utf-8-sig')

    # 4. Graficar
    fig = go.Figure()
    
    # Banda de confianza
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
    
    path_html_out = DIR_INTERMEDIATE / "18_grafico_entropia.html"
    fig.write_html(str(path_html_out))
    
    print(f"💾 Resultados guardados en: {DIR_INTERMEDIATE.name}")
    print("🎉 Abriendo gráfico interactivo en el navegador...")
    webbrowser.open('file://' + str(path_html_out.absolute()))

if __name__ == "__main__":
    main()