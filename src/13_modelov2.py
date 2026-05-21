import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. ARQUITECTURA HÍBRIDA (LOCAL / COLAB)
# ==========================================
try:
    import google.colab
    IN_COLAB = True
    print("☁️ Entorno detectado: Google Colab (Optimizando para GPU)")
except ImportError:
    IN_COLAB = False
    print("💻 Entorno detectado: PC Local (Windows)")

dir_src = Path(__file__).resolve().parent
if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_INTERMEDIATE
    if IN_COLAB:
        DIR_INTERMEDIATE = Path("/content/drive/MyDrive/Datos_Corridos")
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

DIMS = ['intensidad_AR', 'intensidad_MO', 'intensidad_DI', 'intensidad_PO', 'intensidad_NA', 'intensidad_GO']

# ==========================================
# 2. PIPELINE DE ENTRENAMIENTO
# ==========================================
def main():
    print("🚀 [Script 13] Construyendo Modelo V2 Hermenéutico...")
    
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
            df_temp = df_temp.dropna(subset=['es_util_real'])
            dfs_nuevos.append(df_temp)
            
    df_nuevos = pd.concat(dfs_nuevos, ignore_index=True) if dfs_nuevos else pd.DataFrame()
    df_final = pd.concat([df_base, df_nuevos], ignore_index=True)
    
    for dim in DIMS:
        df_final[dim] = pd.to_numeric(df_final[dim], errors='coerce').fillna(0).astype(int)
        
    print(f"📊 Dataset consolidado. Total de estrofas para entrenar: {len(df_final)}")
    
    print("🧠 Calculando embeddings con BGE-M3...")
    modelo_emb = SentenceTransformer('BAAI/bge-m3')
    X = modelo_emb.encode(df_final['texto_entrenamiento'].astype(str).tolist(), show_progress_bar=True)
    Y = df_final[DIMS].values
    
    modelos_v2 = {}
    
    tree_method = 'hist'
    if IN_COLAB:
        try:
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
        
        # --- LA SOLUCIÓN: LabelEncoder Dinámico ---
        # Comprime las clases (ej. [0, 2, 3] -> [0, 1, 2]) de forma transparente
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_dim)
        
        pesos = compute_sample_weight(class_weight='balanced', y=y_encoded)
        
        clf = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            random_state=42
        )
        
        if IN_COLAB and tree_method == 'hist':
            clf.set_params(device='cuda')
            
        # Entrenamos con la versión matemática pura (encoded)
        clf.fit(X, y_encoded, sample_weight=pesos)
        
        # Guardamos AMBOS: El modelo y el decodificador topológico
        modelos_v2[dim] = {
            'clasificador': clf,
            'codificador': le
        }
        print(f"   ✓ Dimensión {dim} mapeó las intensidades {le.classes_} a {np.unique(y_encoded)}")
        
    path_out = DIR_INTERMEDIATE / "multioutput_xgb_models_v2.pkl"
    with open(path_out, 'wb') as f:
        pickle.dump(modelos_v2, f)
        
    print(f"🎉 ¡Modelo V2 completado! Guardado exitosamente en: {path_out}")

if __name__ == "__main__":
    main()