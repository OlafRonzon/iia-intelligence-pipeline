import numpy as np
import pandas as pd
import sys
import os
import pickle
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
import shap
import matplotlib.pyplot as plt

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
    from config import DIR_INTERMEDIATE
    PATH_MODELS_IN = DIR_INTERMEDIATE / "multioutput_xgb_models.pkl" 
    PATH_SHAP_PLOT_PNG = DIR_INTERMEDIATE / "shap_radiografia_barras.png"
    PATH_SHAP_PLOT_HTML = DIR_INTERMEDIATE / "shap_radiografia_texto_colores.html"
except ImportError as e:
    print(f"❌ Error al cargar config.py: {e}")
    sys.exit(1)

MODELO_SOTA = 'BAAI/bge-m3'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Función personalizada para pintar palabras según su valor SHAP
def get_color_from_value(val, max_val):
    if val == 0:
        return "transparent", "black"
    elif val > 0:
        # Rojo para sumar intensidad (Clímax)
        intensity = min(1.0, val / max_val)
        return f"rgba(255, 0, 0, {intensity})", "black" if intensity < 0.6 else "white"
    else:
        # Azul para restar intensidad
        intensity = min(1.0, abs(val) / max_val)
        return f"rgba(0, 100, 255, {intensity})", "black" if intensity < 0.6 else "white"

def generar_explicabilidad_shap():
    print("🚀 Iniciando Módulo 06a: Radiografía con Motor Visual Propio")
    
    if not PATH_MODELS_IN.exists():
        print("❌ No se encontró el archivo de modelos.")
        sys.exit(1)
        
    with open(PATH_MODELS_IN, 'rb') as f:
        modelos = pickle.load(f)
        
    encoder = SentenceTransformer(MODELO_SOTA, device=DEVICE)
    dim_a_auditar = 'AR'
    xgb_model = modelos[dim_a_auditar]
    
    def predict_text_probability(texts):
        embeddings = encoder.encode(texts, show_progress_bar=False, device=DEVICE, convert_to_numpy=True)
        return xgb_model.predict_proba(embeddings)

    masker = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.Explainer(predict_text_probability, masker=masker, output_names=["Int. 0", "Int. 1", "Int. 2", "Int. 3"])
    
    verso_ejemplo = "Traigo un cuerno de chivo bien terciado y chaleco antibalas con mi gente coordinando"
    print(f"🔬 Calculando matemática SHAP...")
    shap_values = explainer([verso_ejemplo])
    
    # ---------------------------------------------------------
    # Gráfico de Barras Estático (Esto ya funcionaba bien)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values[0, :, "Int. 3"], show=False)
    plt.title(f"Importancia para Intensidad 3 - Dimensión {dim_a_auditar}")
    plt.savefig(str(PATH_SHAP_PLOT_PNG), bbox_inches='tight', dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # NUEVO: Construcción manual del HTML coloreado
    # ---------------------------------------------------------
    palabras = shap_values.data[0]
    valores_clase_3 = shap_values.values[0, :, 3] # Extraemos los valores matemáticos de la intensidad 3
    
    # Encontramos el valor máximo para escalar los colores
    max_abs_val = max(abs(valores_clase_3.min()), valores_clase_3.max())
    if max_abs_val == 0: max_abs_val = 0.0001
    
    html_spans = []
    for palabra, valor in zip(palabras, valores_clase_3):
        bg_color, text_color = get_color_from_value(valor, max_abs_val)
        # Limpiamos saltos de línea para que se vea bien
        palabra_limpia = palabra.replace("\n", " ")
        span = f'<span style="background-color: {bg_color}; color: {text_color}; padding: 2px 4px; border-radius: 4px; margin: 0 1px; display: inline-block;">{palabra_limpia}</span>'
        html_spans.append(span)
        
    verso_coloreado = "".join(html_spans)
    
    html_wrapper = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Radiografía del Deseo (Personalizada)</title>
    </head>
    <body style="background-color: #f8f9fa; padding: 50px; font-family: 'Segoe UI', sans-serif; text-align: center;">
        <h2 style="color: #333;">Radiografía del Agenciamiento</h2>
        <p style="color: #666; margin-bottom: 30px;">
            <span style="background-color: rgba(255,0,0,0.7); color: white; padding: 2px 5px; border-radius: 3px;">Rojo</span> empuja hacia el Clímax (Intensidad 3). 
            <span style="background-color: rgba(0,100,255,0.7); color: white; padding: 2px 5px; border-radius: 3px;">Azul</span> lo aleja.
        </p>
        
        <div style="font-size: 24px; line-height: 2; padding: 30px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; max-width: 800px;">
            {verso_coloreado}
        </div>
    </body>
    </html>
    """
    
    with open(PATH_SHAP_PLOT_HTML, "w", encoding="utf-8") as f:
        f.write(html_wrapper)
        
    print(f"✅ Radiografía HTML forzada y coloreada generada: {PATH_SHAP_PLOT_HTML}")

if __name__ == "__main__":
    generar_explicabilidad_shap()