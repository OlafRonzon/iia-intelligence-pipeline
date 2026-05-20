import pandas as pd
import numpy as np
import sys
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import torch

# ==========================================
# 1. CONEXIÓN UNIVERSAL Y DIRECTORIOS
# ==========================================
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    dir_src = Path(os.getcwd()) / "src"

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_VALIDATION, DICCIONARIO_PENTADIMENSIONAL
    DIR_OUTPUT = dir_src.parent / "visualizaciones"
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

# Stopwords básicas para limpiar la síntesis léxica
STOPWORDS_ES = set("el la los las un una unos unas y e o u a ante bajo cabe con contra de desde en entre hacia hasta para por segun sin so sobre tras que porque como cuando donde quien cuyo cual al del se su sus mi mis tu tus te lo le me nos os es son ser esta este estos estas era fue han ha habia".split())

# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN (ESTRATOS)
# ==========================================
def cargar_estratos():
    """Carga y diferencia los artefactos Regex (validados) y Vectoriales (deriva)."""
    print("🔍 [1/4] Excavando estratos arqueológicos (Regex vs Vectorial)...")
    
    df_regex = pd.DataFrame()
    df_vector = pd.DataFrame()
    
    for dimension in DICCIONARIO_PENTADIMENSIONAL.keys():
        # 1. Cargar Regex (Ground Truth)
        path_regex = DIR_VALIDATION / f"5_booster_{dimension}.csv"
        if path_regex.exists():
            df_temp = pd.read_csv(path_regex)
            col_val = f"val_{dimension}"
            if col_val in df_temp.columns:
                df_temp[col_val] = pd.to_numeric(df_temp[col_val], errors='coerce').fillna(0)
                # Filtramos SOLO los validados con intensidad > 0
                df_temp = df_temp[df_temp[col_val] > 0].copy()
                df_temp['Origen'] = 'Regex (Ancla Semántica)'
                df_regex = pd.concat([df_regex, df_temp], ignore_index=True)

        # 2. Cargar Vectorial (La Máquina Deseante sin validar)
        path_vector = DIR_VALIDATION / f"5_booster_{dimension}_vectorial.csv"
        if path_vector.exists():
            df_temp = pd.read_csv(path_vector)
            df_temp['Origen'] = 'Vectorial (Deriva Latente)'
            df_vector = pd.concat([df_vector, df_temp], ignore_index=True)

    # Limpiar años (Cronología)
    df_total = pd.concat([df_regex, df_vector], ignore_index=True)
    df_total['year'] = pd.to_numeric(df_total['year'], errors='coerce')
    df_total = df_total.dropna(subset=['year', 'verso_texto'])
    
    print(f"   ✅ Anclas recuperadas (Regex): {len(df_regex)}")
    print(f"   ✅ Deriva recuperada (Vectorial): {len(df_vector)}")
    
    return df_total

# ==========================================
# 3. MÓDULOS DE CARTOGRAFÍA VISUAL
# ==========================================
def cartografia_cronologica(df):
    """Genera el gráfico de densidad temporal."""
    print("⏳ [2/4] Trazando la intensidad temporal...")
    plt.figure(figsize=(12, 6))
    
    sns.kdeplot(data=df, x="year", hue="Origen", fill=True, common_norm=False, 
                palette={"Regex (Ancla Semántica)": "#2c3e50", "Vectorial (Deriva Latente)": "#e74c3c"},
                alpha=0.5, linewidth=2)
    
    plt.title("Densidad Cronológica del Agenciamiento Narco", fontsize=16, fontweight='bold')
    plt.xlabel("Año", fontsize=12)
    plt.ylabel("Intensidad (Densidad de Versos)", fontsize=12)
    plt.xlim(1970, 2026)
    
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "01_cronologia_agenciamiento.png", dpi=300)
    plt.close()

def sintesis_semantica(df):
    """Filtra palabras únicas que descubrió el algoritmo vectorial y no el Regex."""
    print("🗣️ [3/4] Destilando el léxico de la máquina deseante...")
    
    textos_regex = " ".join(df[df['Origen'] == 'Regex (Ancla Semántica)']['verso_texto'].astype(str)).lower()
    textos_vector = " ".join(df[df['Origen'] == 'Vectorial (Deriva Latente)']['verso_texto'].astype(str)).lower()
    
    # Tokenización simple
    palabras_regex = set(re.findall(r'\b[a-zñáéíóú]+\b', textos_regex))
    tokens_vector = re.findall(r'\b[a-zñáéíóú]+\b', textos_vector)
    
    # Fuga semántica: palabras en vector que NO existen en regex ni en stopwords
    fuga = [w for w in tokens_vector if w not in palabras_regex and w not in STOPWORDS_ES and len(w) > 3]
    
    # Conteo de frecuencias de la fuga
    frecuencias = pd.Series(fuga).value_counts().head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=frecuencias.values, y=frecuencias.index, palette="viridis")
    plt.title("Léxico de Fuga (Top 20 palabras exclusivas de la síntesis Vectorial)", fontsize=14)
    plt.xlabel("Frecuencia de aparición")
    plt.ylabel("Términos descubiertos")
    
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "02_fuga_semantica.png", dpi=300)
    plt.close()

def cartografia_topologica(df):
    """Proyecta el espacio plegado de 1024 dimensiones a 2D usando t-SNE."""
    print("🌌 [4/4] Proyectando el espacio vectorial plegado (t-SNE)...")
    
    # Cargar el modelo SOTA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    
    textos = df['verso_texto'].tolist()
    print("   🧠 Calculando tensores para proyección espacial...")
    embeddings = model.encode(textos, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    
    # Reducción de dimensionalidad (t-SNE)
    # Perplexity ajustada matemáticamente al tamaño de la muestra
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    
    print("   🗺️ Desplegando dimensiones (esto puede tardar unos segundos)...")
    vectores_2d = tsne.fit_transform(embeddings)
    
    df['x'] = vectores_2d[:, 0]
    df['y'] = vectores_2d[:, 1]
    
    # Graficar
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='Origen', style='Origen',
                    palette={"Regex (Ancla Semántica)": "#2980b9", "Vectorial (Deriva Latente)": "#c0392b"},
                    s=80, alpha=0.7, edgecolor='w')
    
    plt.title("Topología del Deseo: Anclas Teóricas vs Deriva Latente", fontsize=16, fontweight='bold')
    plt.xlabel("Dimensión Topológica 1")
    plt.ylabel("Dimensión Topológica 2")
    plt.legend(title="Estrato de Origen")
    
    # Quitar bordes para que se vea como un espacio flotante
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "03_espacio_plegado_tsne.png", dpi=300)
    plt.close()

# ==========================================
# 4. EJECUCIÓN DEL FLUJO
# ==========================================
def main():
    print("==================================================")
    print(" INICIANDO CARTOGRAFÍA VISUAL NO TELEOLÓGICA")
    print("==================================================")
    
    df_maestro = cargar_estratos()
    
    if df_maestro.empty:
        print("❌ No hay datos suficientes para graficar. Revisa que existan los archivos en DIR_VALIDATION.")
        return
        
    cartografia_cronologica(df_maestro)
    sintesis_semantica(df_maestro)
    cartografia_topologica(df_maestro)
    
    print("\n✅ Arqueología completada. Revisa la carpeta 'visualizaciones' para ver los mapas empíricos.")

if __name__ == "__main__":
    main()