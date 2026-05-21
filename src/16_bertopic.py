import os
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

def main():
    print("🧬 [Paso 17] Iniciando Extracción de ADN Semántico con BERTopic...")
    
    # Rutas en Colab
    path_definitivo = Path("/content/drive/MyDrive/Datos_Corridos/CORPUS_DEFINITIVO_TESIS.csv")
    path_raw = Path("/content/drive/MyDrive/Datos_Corridos/letras_corpus_final.csv")
    dir_out = Path("/content/drive/MyDrive/Datos_Corridos/data/02_intermediate")
    
    # 1. Recuperar las letras de las canciones filtradas
    df_filtrado = pd.read_csv(path_definitivo)
    df_crudo = pd.read_csv(path_raw)
    
    # Cruzar para obtener el texto (lyrics)
    df = pd.merge(df_filtrado, df_crudo[['artist', 'song', 'lyrics']], on=['artist', 'song'], how='inner')
    df = df.dropna(subset=['lyrics'])
    textos = df['lyrics'].tolist()
    
    print(f"✅ Textos recuperados: {len(textos)} canciones puras de agenciamiento.")

    # 2. Configurar la Reducción de Dimensionalidad (UMAP) y Clustering (HDBSCAN)
    # min_cluster_size=15 obliga al modelo a no hacer micro-tópicos
    # min_samples=5 relaja la restricción de ruido (bloqueando que todo sea -1)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    
    modelo_emb = SentenceTransformer('BAAI/bge-m3')

    topic_model = BERTopic(
        embedding_model=modelo_emb,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="spanish",
        calculate_probabilities=False
    )

    # 3. Entrenar y extraer Tópicos
    print("🧠 Calculando clústeres semánticos (Esto tomará un par de minutos)...")
    topics, probs = topic_model.fit_transform(textos)
    
    # 4. Extraer resultados para ti
    df_info = topic_model.get_topic_info()
    path_csv = dir_out / "17_bertopic_resumen.csv"
    df_info.to_csv(path_csv, index=False, encoding='utf-8-sig')
    print(f"💾 Resumen de Tópicos guardado en: {path_csv.name}")

    # 5. Generar Visualización de Alta Calidad
    fig = topic_model.visualize_barchart(top_n_topics=12, n_words=8, title="ADN Semántico: Tópicos Latentes en el Agenciamiento")
    fig.update_layout(template="plotly_dark")
    
    path_html = dir_out / "17_grafico_topicos.html"
    fig.write_html(str(path_html))
    print(f"📊 Dashboard exportado a: {path_html.name}")

if __name__ == "__main__":
    main()