import os
import sys
import pickle
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. ARQUITECTURA HÍBRIDA (LOCAL / COLAB)
# ==========================================
# Detectar entorno
try:
    import google.colab
    IN_COLAB = True
    print("☁️ Entorno detectado: Google Colab (Optimizando para GPU T4)")
except ImportError:
    IN_COLAB = False
    print("💻 Entorno detectado: PC Local (Windows)")

# Resolución de rutas relativas absolutas
dir_src = Path(__file__).resolve().parent
if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    # Importamos desde tu config.py (funciona igual en local o clonado en Colab)
    from config import DIR_INTERMEDIATE
    
    # OVERRIDE PARA COLAB: Si tus datos pesados viven en Drive y no en el repo clonado
    if IN_COLAB:
        DIR_INTERMEDIATE = Path("/content/drive/MyDrive/Datos_Corridos/data/02_intermediate")
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

DIMS = ['intensidad_AR', 'intensidad_MO', 'intensidad_DI', 'intensidad_PO', 'intensidad_NA', 'intensidad_GO']

# ==========================================
# 2. PIPELINE DE ENTRENAMIENTO
# ==========================================
def main():
    print("🚀 [Script 13] Construyendo Modelo V2 Hermenéutico...")
    
    # Cargar y fusionar los datos
    path_base = DIR_INTERMEDIATE / "gold_standard_contexto_expandido.csv"
    path_12 = DIR_INTERMEDIATE / "12_estrofas_para_validar.csv"
    path_12a = DIR_INTERMEDIATE / "12_a_estrofas_recientes.csv"
    
    try:
        df_base = pd.read_csv(path_base, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"❌ No se encontró el dataset base en: {path_base}")
        return

    dfs_nuevos = []
    for p in [path_12, path_12a]:
        if p.exists():
            df_temp = pd.read_csv(p, encoding='utf-8-sig')
            # Filtrar solo las que validaste (ignorando vacías)
            df_temp = df_temp.dropna(subset=['es_util_real'])
            dfs_nuevos.append(df_temp)
            
    df_nuevos = pd.concat(dfs_nuevos, ignore_index=True) if dfs_nuevos else pd.DataFrame()
    df_final = pd.concat([df_base, df_nuevos], ignore_index=True)
    
    # Forzar tipado numérico estricto para XGBoost
    for dim in DIMS:
        df_final[dim] = pd.to_numeric(df_final[dim], errors='coerce').fillna(0).astype(int)
        
    print(f"📊 Dataset consolidado. Total de estrofas para entrenar: {len(df_final)}")
    
    # Vectorización Densa (Acelerada por GPU si estás en Colab)
    print("🧠 Calculando embeddings con BGE-M3...")
    modelo_emb = SentenceTransformer('BAAI/bge-m3')
    
    # sentence-transformers usará 'cuda' automáticamente si estás en Colab con T4
    X = modelo_emb.encode(df_final['texto_entrenamiento'].astype(str).tolist(), show_progress_bar=True)
    Y = df_final[DIMS].values
    
    # Entrenamiento Multi-Salida con Rigor Matemático
    modelos_v2 = {}
    
    # XGBoost: 'hist' usa CPU, 'gpu_hist' (o 'hist' con device='cuda') usa GPU
    tree_method = 'hist'
    if IN_COLAB:
        try:
            # XGBoost > 2.0 usa device='cuda', versiones previas usan tree_method='gpu_hist'
            import xgboost
            if int(xgboost.__version__.split('.')[0]) >= 2:
                tree_method = 'hist'
            else:
                tree_method = 'gpu_hist'
        except:
            pass

    for idx, dim in enumerate(DIMS):
        print(f"⚙️ Entrenando clasificador topológico: {dim}...")
        y_dim = Y[:, idx]
        
        # Penalización asimétrica de clases
        pesos = compute_sample_weight(class_weight='balanced', y=y_dim)
        
        clf = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            random_state=42
        )
        
        # Si tienes XGBoost 2.0+ en colab, forzamos CUDA
        if IN_COLAB and tree_method == 'hist':
            clf.set_params(device='cuda')
            
        clf.fit(X, y_dim, sample_weight=pesos)
        modelos_v2[dim] = clf
        
    # Guardar el Nuevo Cerebro
    path_out = DIR_INTERMEDIATE / "multioutput_xgb_models_v2.pkl"
    with open(path_out, 'wb') as f:
        pickle.dump(modelos_v2, f)
        
    print(f"🎉 ¡Modelo V2 completado! Guardado exitosamente en: {path_out}")

if __name__ == "__main__":
    main()