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

# 3. EL FILTRO DE STOP WORDS (LA MAGIA MEJORADA AQUÍ)
    print("🧹 Preparando filtro de palabras vacías y jerga...")
    
    # Obtenemos las stopwords de NLTK
    stop_words_es = stopwords.words('spanish')
    
    # Agregamos muletillas propias de la música regional
    muletillas_regionales = ['pa', 'ahí', 'si', 'pos', 'bien', 'ya', 'así', 'mas', 'ay', 'voy', 'va', 'pues']
    
    # Unimos todo en una sola lista
    todas_las_stopwords = stop_words_es + muletillas_regionales

    # TRUCO: Quitamos acentos de las stopwords para que el filtro sea infalible
    import unicodedata
    def limpiar_acentos(texto):
        return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    
    stopwords_limpias = [limpiar_acentos(palabra).lower() for palabra in todas_las_stopwords]
    
    # Aseguramos explícitamente a los "rebeldes" que vimos en tu gráfica
    stopwords_limpias.extend(['la', 'los', 'el', 'de', 'las', 'un', 'una', 'en', 'que', 'me', 'se', 'su', 'mi'])

    # Configuramos el vectorizador obligándolo a ignorar acentos en las letras
    vectorizer_model = CountVectorizer(
        stop_words=stopwords_limpias, 
        ngram_range=(1, 2),
        strip_accents='unicode' # <--- CLAVE PARA QUE COINCIDAN LOS TEXTOS
    )

    # 4. Configurar Modelos
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean', cluster_selection_method='eom')
    modelo_emb = SentenceTransformer('BAAI/bge-m3')

    print("🤖 Configurando BERTopic...")
    topic_model = BERTopic(
        embedding_model=modelo_emb,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False
    )

    # --- NUEVA RED DE SEGURIDAD ---
    # Verificamos que realmente tengamos letras para analizar
    cantidad_textos = len(textos)
    print(f"Letras listas para procesar: {cantidad_textos}")

    if cantidad_textos == 0:
        print("⚠️ ALERTA: No hay canciones para analizar. Revisa el filtro de densidad semántica.")
    else:
        print("🧠 Entrenando el modelo (haciendo fit)... Esto puede tomar unos minutos.")
        # ESTA ES LA LÍNEA CRÍTICA QUE RESUELVE TU ERROR:
        topics, probs = topic_model.fit_transform(textos)
        
        # 5. Exportar Resultados
        print("💾 Guardando resultados...")
        path_csv = dir_base / "17_bertopic_resumen.csv"
        
        # Como ya hicimos fit_transform, get_topic_info() funcionará perfectamente
        topic_model.get_topic_info().to_csv(path_csv, index=False, encoding='utf-8-sig')
        
        # Generar gráfica de barras
        fig = topic_model.visualize_barchart(top_n_topics=12, n_words=8, title="ADN Semántico: Tópicos Latentes (Limpio)")
        fig.update_layout(template="plotly_dark")
        
        path_html = dir_base / "17_grafico_topicos.html"
        fig.write_html(str(path_html))
        print(f"📊 Dashboard exportado. ¡Descarga y ábrelo!")
    
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