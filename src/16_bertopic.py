import os
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Descargar las stopwords de español
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def main():
    print("🧬 [Paso 17] Iniciando Extracción de ADN Semántico con Limpieza...")
    
    # 1. Rutas
    dir_base = Path("/content/drive/MyDrive/Datos_Corridos")
    path_inferido = dir_base / "14_corpus_10k_inferido_v2.csv"
    path_raw = dir_base / "corpus_final_agrupado.csv" 
    
    if not path_inferido.exists() or not path_raw.exists():
        print("❌ Faltan archivos en el Drive.")
        return

    # 2. Cruce de datos
    print("📖 Cruzando los ganadores con sus letras originales...")
    df_10k = pd.read_csv(path_inferido, encoding='utf-8-sig')
    df_crudo = pd.read_csv(path_raw, encoding='utf-8-sig')
    
    df_filtrado = df_10k[df_10k['densidad_semantica_v2'] >= 0.0333]
    df_final = pd.merge(df_filtrado, df_crudo[['artist', 'song', 'lyrics']], on=['artist', 'song'], how='inner')
    df_final = df_final.dropna(subset=['lyrics'])
    
    textos = df_final['lyrics'].tolist()
    print(f"✅ Se armó el corpus en memoria con {len(textos)} letras completas.")

    # 3. EL FILTRO DE STOP WORDS (LA MAGIA AQUÍ)
    print("🧹 Preparando filtro de palabras vacías y jerga...")
    stop_words_es = stopwords.words('spanish')
    # Agregamos muletillas propias de la música regional
    muletillas_regionales = ['pa', 'ahí', 'si', 'pos', 'bien', 'ya', 'así', 'mas', 'ay', 'voy', 'va', 'pues']
    stop_words_es.extend(muletillas_regionales)
    
    vectorizer_model = CountVectorizer(stop_words=stop_words_es, ngram_range=(1, 2))

    # 4. Configurar Modelos
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    modelo_emb = SentenceTransformer('BAAI/bge-m3')

    print("🤖 Extrayendo los clústeres latentes...")
    topic_model = BERTopic(
        embedding_model=modelo_emb,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model, # <--- INYECTAMOS EL FILTRO AQUÍ
        language="spanish",
        calculate_probabilities=False
    )

    topics, probs = topic_model.fit_transform(textos)
    
    # 5. Exportar Resultados
    path_csv = dir_base / "17_bertopic_resumen.csv"
    topic_model.get_topic_info().to_csv(path_csv, index=False, encoding='utf-8-sig')
    
    # Generar gráfica de barras
    fig = topic_model.visualize_barchart(top_n_topics=12, n_words=8, title="ADN Semántico: Tópicos Latentes (Limpio)")
    fig.update_layout(template="plotly_dark")
    
    path_html = dir_base / "17_grafico_topicos.html"
    fig.write_html(str(path_html))
    print(f"📊 Dashboard exportado. ¡Descarga y ábrelo!")

if __name__ == "__main__":
    main()