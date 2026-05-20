import numpy as np
import pandas as pd
import sys
import os
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
try:
    dir_src = Path(__file__).resolve().parent
except NameError:
    dir_src = Path(os.getcwd()) / "src"

if str(dir_src) not in sys.path:
    sys.path.append(str(dir_src))

try:
    from config import DIR_INTERMEDIATE, PATH_GOLD_STANDARD_VECTORS, PATH_GOLD_STANDARD_LABELS, DICCIONARIO_PENTADIMENSIONAL
    # Guardaremos esta versión sobreescribiendo el archivo anterior
    PATH_MODELS_OUT = DIR_INTERMEDIATE / "multioutput_xgb_models.pkl"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

def entrenar_pipeline_balanceado():
    print("🚀 Iniciando Fase 3 V2.0: Entrenamiento del Motor Multi-Etiqueta (Con Balanceo de Pesos)")
    
    if not PATH_GOLD_STANDARD_VECTORS.exists() or not PATH_GOLD_STANDARD_LABELS.exists():
        print("❌ No se encontraron las matrices del Gold Standard. Corre la Fase 1 primero.")
        sys.exit(1)
        
    X = np.load(PATH_GOLD_STANDARD_VECTORS)
    Y = np.load(PATH_GOLD_STANDARD_LABELS)
    
    # Estratificación Multi-Etiqueta (Hashing de Filas)
    string_labels = ["_".join(map(str, row)) for row in Y]
    df_strat = pd.DataFrame({'strat_key': string_labels})
    counts = df_strat['strat_key'].value_counts()
    df_strat['strat_key'] = df_strat['strat_key'].apply(lambda x: x if counts[x] > 1 else 'rare_combination')
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42, stratify=df_strat['strat_key']
    )
    
    dimensiones = list(DICCIONARIO_PENTADIMENSIONAL.keys())
    modelos_entrenados = {}
    
    for idx, dim in enumerate(dimensiones):
        print(f"\n🧠 Entrenando Clasificador para Dimensión: {dim}...")
        
        y_train_dim = Y_train[:, idx].astype(int)
        y_test_dim = Y_test[:, idx].astype(int)
        
        # ---------------------------------------------------------
        # ⚖️ EL MOTOR DE RIGOR METODOLÓGICO: Penalización por Frecuencia Inversa
        # ---------------------------------------------------------
        # Calcula cuánto pesa cada verso. Si la intensidad 3 es rara, su peso matemático será altísimo.
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_dim)
        
        # Redujimos el learning_rate y bajamos max_depth para evitar sobreajustar con los nuevos pesos
        model = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            early_stopping_rounds=50,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Inyectamos los pesos en la función de costo
        model.fit(
            X_train, y_train_dim,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test_dim)],
            verbose=False
        )
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test_dim, preds)
        print(f"📈 Dimensión {dim} - Accuracy Ponderado: {acc:.4f} (Árboles óptimos: {model.best_iteration})")
        print(classification_report(y_test_dim, preds, zero_division=0))
        
        modelos_entrenados[dim] = model
        
    with open(PATH_MODELS_OUT, 'wb') as f:
        pickle.dump(modelos_entrenados, f)
        
    print(f"\n🎉 Fase 3 V2 completada con Rigor Matemático. Modelos guardados en: {PATH_MODELS_OUT}")

if __name__ == "__main__":
    entrenar_pipeline_balanceado()