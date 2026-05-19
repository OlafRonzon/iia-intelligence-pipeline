from pathlib import Path

# ==========================================
# 1. GESTIÓN DE RUTAS (PATHS) RELATIVAS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas de datos
DIR_RAW = BASE_DIR / "data" / "01_raw"
DIR_INTERMEDIATE = BASE_DIR / "data" / "02_intermediate"
DIR_PROCESSED = BASE_DIR / "data" / "03_processed"
DIR_VALIDATION = BASE_DIR / "validacion_manual"

# Alias para compatibilidad de scripts
INT_DIR = DIR_INTERMEDIATE

# Rutas exactas de archivos (Pipeline secuencial)
PATH_CORPUS_CRUDO = DIR_RAW / "letras_corpus_final.csv"
PATH_CORPUS_MARCADO = INT_DIR / "1_corpus_marcado_narco.csv"
PATH_AUDITORIA_CONTEXTOS = INT_DIR / "2_auditoria_contextos.csv"

# Salidas del Clustering (Script 03)
PATH_CORPUS_AGRUPADO = INT_DIR / "2_corpus_agrupado_ia.csv"
PATH_CORPUS_FILTRADO_CLUSTERS = INT_DIR / "3_corpus_pre_muestreo.csv"

# Salidas de Validación (Script 04 y 05)
PATH_MUESTRA_CONTROL = DIR_VALIDATION / "muestra_control.csv"
PATH_FILTRO_FINAL = DIR_PROCESSED / "10k_filtrado_semantico.csv"

#salida de 03 
# Nueva ruta de salida para los versos
PATH_VERSOS_ATOMIZADOS = DIR_INTERMEDIATE / "3_versos_scoring_bruto.csv"

#salida de 04 
PATH_ETIQUETAS_PROVISIONALES = DIR_INTERMEDIATE / "4_versos_etiquetados_snorkel.csv"

#paths etapa 05
# Dentro de config.py
PATH_TAREAS_VALIDACION = DIR_VALIDATION / "5_tareas_pendientes_gs.csv"
PATH_GOLD_STANDARD = DIR_VALIDATION / "5_gold_standard_final.csv"
 #==========================================
# 2. PARÁMETROS DEL PIPELINE
# ==========================================
# Control de Calidad y Muestreo
UMBRAL_MINIMO_PALABRAS = 2      # Mínimo de aciertos para marcar como "narco"
VENTANA_CONTEXTO = 4            # Palabras a los lados para el análisis de frases
MUESTRAS_POR_DECADA = 10        # Lo que TÚ vas a leer para validar
TOP_CANCIONES_POR_DECADA = 50   # Límite de ranking por década para el script 02

# Filtros Semánticos
UMBRAL_RECHAZO_SEMANTICO = 0.5  # Sensibilidad de la similitud de coseno
NUM_CLUSTERS = 15               # Grupos para K-Means
# ==========================================
# 3. DICCIONARIOS Y LÉXICOS
# ==========================================
PALABRAS_IRRELEVANTES = [ "que", "de", "y", "la", "no", "a", "me", "el", "te", "en", "mi", "se", "yo", "por", "lo", "un", "es", "con", "si", "ya", "los", "las", "para", "al", "como", "pero", "una", "le", "que",
"de",
"y",
"la",
"no",
"a",
"me",
"el",
"te",
"en",
"mi",
"se",
"yo",
"por",
"lo",
"un",
"es",
"tu",
"con",
"si",
"ya",
"los",

"las",
"para",
"al",
"como",
"pero",
"una",
"le",
"su",
"tú",
"porque",
"cuando",
"del",
"más",
"sin",
"voy",
"ti",
"soy",
"pa",
"muy",
"qué",
"mis",
"ni",
"solo",
"ay",
"hay",
"bien",
"esta",
"mí",
"tus",
"eso",
"así",
"he",
"nos",
"vez",
"este",
"tan",

"fue",
"o",
"eres",
"va",
"hasta",
"día",
"aquí",
"sé",
"ha",
"donde",
"mas",
"les",
"aunque",
"vas",
"hoy",




"dos",

"sus",







"está",


"también",
"pues",

"oh",

"ese",

"cómo",
"ahí",



"mejor",






"hace",


"dicen",



"lado",





"desde",
"has",

"hacer",
"sí",
"cada",



"van",



"estás",
"the",
"adiós",




"sabes",
"han",
"puede",




"you",
"dijo",
"ando",



"allá",


"digo",
"paso",
"importa",
"quién",
"vamos",
"tal",
"i",
"quien",
"entre",








"ah",




"mira",
"estar",

"esto",



"eh",

"dice",
"da",




"sea",

"mano",

"ja",




"di",

"tres"
    # Ejemplos: "que", "de", "y", "a", "la", "el", "en"
]

SET_NARCO_FUERTE = {
    # Armas
    "cuerno", "ak-47", "r15", "fusil", "pistola", "granada", "blindaje",
    # Drogas
    "cocaína", "perico", "mota", "cristal", "fentanilo", "polvo", "hierba", "gallo","blunt", "carga",
    # Movilidad
    "troca", "blindada", "camioneta", "avioneta", "suburban","calle", "caballo", "mercancia", "mercancía", "policía",
    "barrio", "sierra", "tierra",
    # Jerarquía / Organización
    "cártel", "patrón", "jefe", "sicario", "teniente", "comandante",
    "plaza", "nómina", "halcón", "puntero","hombres", "nivel", "grandeza", "dolar", "millones","ley", "defender", "conectas", "chapo", "chapiza", "distribuir", "distribuirla", "gatilleros",
    "compa", "banda", "real",
    # Ostentación / Violencia
    "belicón", "bélico", "alterado", "encapuchado", "ejecutado",
    "levantón", "encajuelado", "descuartizar", "sanguinario",
    "gente", "dinero", "muerte", "mujeres","cuatro", "tiros", "muerto",
    "balazos", "frente", 
    "miedo", "carro", "mal", "matar", "waxito", "galliza", "wax", "contrabando", "dólares", "dolares", "tira", "judiciales", "federales", "traficantes",
    # Cultura
    "corrido", "chota","valientes", "malos", "perdió", "perdio", "gobierno", "azteca", "billetes", "diablo" 
}
DICCIONARIO_PENTADIMENSIONAL = {
    "AR": [ # Armas y Violencia
        "cuerno", "ak-47", "r15", "fusil", "pistola", "granada", "blindaje", "muerto", "encapuchado", "cuatro", "defender", "malos", "tirar", "traficantes",
        "tiros", "balazos", "ejecutado", "levantón", "matar", "muerte", "sanguinario", "gatilleros", "gatillero", "encajuelado", "descuartizar", "belicón", "bélico",
        "m16","calibre", "rifle", "arma", "armas"
    ],
    "MO": [ # Movilidad y Blindaje
        "troca", "blindada", "camioneta", "avioneta", "suburban", "carro", "mercancia", "mercancía", "producto", "kilos", "carga", "kilo",
        "camino", "sierra", "frontera", "ruta", "flete", "avioneta", "tunel", "tuneles", "caballo", "contrabando"
    ], #se eligieron palabras relacionados con logística, transporte y producto
    "DI": [ # Ostentación y Dinero
        "dinero", "dólares", "dolares", "billetes", "pacas", "lujos", "marcas",
        "millones", "riqueza", "joyas", "diamantes", "oro", "mujeres"
    ],
    "PO": [ # Poder Organizacional
        "cártel", "patrón", "jefe", "sicario", "teniente", "comandante", "gente", "nivel", "grandeza", "chapo", "chapiza", "hombres", "banda",
        "plaza", "nómina", "halcón", "puntero", "mando", "clave", "orden", "señor", "barrio"
    ],
    "NA": [ # Consumo y Narcóticos
        "cocaína", "perico", "mota", "cristal", "fentanilo", "polvo", "hierba", "bong", "blunt", "gallo", 
        "carga", "mercancía", "kilo", "cocina", "clavo" , "waxito", "galliza", "wax", "coca", "meta", "marihuana", "mariguana", "mariguana", "mariguana", "marijuana", "mariguana"
    ],
    "GO": [ # Gobierno y Legalidad 
        "chota", "gobierno", "americano", "federales", "judiciales", "tira", "policía", "ley", "puercos"]
}