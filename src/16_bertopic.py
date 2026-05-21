import os
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

def main():
    print("🧬 [Paso 17] Iniciando Extracción de ADN Semántico con BERTopic...")
    
    # 1. Ruta plana directa a tu carpeta en Drive
    dir_base = Path("/content/drive/MyDrive/Datos_Corridos")
    
    path_inferido = dir_base / "14_corpus_10k_inferido_v2.csv"
    path_raw = dir_base / "corpus_final_agrupado.csv" 
    
    if not path_inferido.exists():
        print(f"❌ No encuentro: {path_inferido}")
        return
    if not path_raw.exists():
        print(f"❌ No encuentro: {path_raw}")
        return

    # 2. El "BuscarV" rápido para armar las letras
    print("📖 Cruzando los ganadores con sus letras originales...")
    df_10k = pd.read_csv(path_inferido, encoding='utf-8-sig')
    df_crudo = pd.read_csv(path_raw, encoding='utf-8-sig')
    
    # Filtramos las ~1300 ganadoras
    UMBRAL = 0.0333
    df_filtrado = df_10k[df_10k['densidad_semantica_v2'] >= UMBRAL]
    
    # Pegamos la columna 'lyrics' original a las canciones ganadoras
    df_final = pd.merge(df_filtrado, df_crudo[['artist', 'song', 'lyrics']], on=['artist', 'song'], how='inner')
    df_final = df_final.dropna(subset=['lyrics'])
    
    textos = df_final['lyrics'].tolist()
    print(f"✅ ¡Éxito! Se armó el corpus en memoria con {len(textos)} letras completas.")

    if len(textos) == 0:
        return

    # 3. BERTopic (Solo leerá las canciones filtradas)
    print("🤖 Extrayendo los clústeres latentes...")
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

    topics, probs = topic_model.fit_transform(textos)
    
    # 4. Guardar resultados directamente en Datos_Corridos
    path_csv = dir_base / "17_bertopic_resumen.csv"
    topic_model.get_topic_info().to_csv(path_csv, index=False, encoding='utf-8-sig')
    print(f"💾 Resumen guardado en: {path_csv.name}")

    fig = topic_model.visualize_barchart(top_n_topics=12, n_words=8, title="ADN Semántico: Tópicos Latentes")
    fig.update_layout(template="plotly_dark")
    
    path_html = dir_base / "17_grafico_topicos.html"
    fig.write_html(str(path_html))
    print(f"📊 Gráfico guardado en: {path_html.name}")

if __name__ == "__main__":
    main()