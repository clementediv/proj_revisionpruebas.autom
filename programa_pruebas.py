import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# -------------------------
# 1) Cargar XML de respuestas y pauta
# -------------------------
tree_res = ET.parse("respuestas_examen.xml")
root_res = tree_res.getroot()

tree_key = ET.parse("pauta_examen.xml")
root_key = tree_key.getroot()
pauta = [a.strip().upper() for a in root_key.find("respuestas").text.split(",")]

num_preg = len(pauta)

# -------------------------
# 2) Construir DataFrame de respuestas (letras) y de aciertos (1/0/NaN)
# -------------------------
alumnos = []
mat_res_letras = []

for est in root_res.findall("estudiante"):
    alumnos.append(est.get("id"))
    letras = [x.strip().upper() for x in est.find("respuestas").text.split(",")]
    # Si vienen más/menos respuestas, ajustamos a num_preg
    if len(letras) < num_preg:
        letras += [""] * (num_preg - len(letras))
    elif len(letras) > num_preg:
        letras = letras[:num_preg]
    mat_res_letras.append(letras)

df_letras = pd.DataFrame(mat_res_letras, index=alumnos, columns=[f"Q{i+1}" for i in range(num_preg)])

# Matriz de correctas (1), incorrectas (0), omitidas (NaN)
def letra_a_score(letra, key):
    if letra == "" or letra not in {"A","B","C","D"}:
        return np.nan  # omitida
    return 1.0 if letra == key else 0.0

df_score = pd.DataFrame(
    {
        f"Q{i+1}": [letra_a_score(df_letras.iloc[r, i], pauta[i]) for r in range(len(df_letras))]
        for i in range(num_preg)
    },
    index=alumnos
)

# Puntaje total por alumno (tratando NaN como 0 para el total)
total = df_score.fillna(0).sum(axis=1)

# -------------------------
# 3) Funciones auxiliares para métricas por ítem
# -------------------------
def indice_discriminacion(col_scores, total_scores, top_frac=0.27):
    # col_scores: Serie con 1/0/NaN del ítem
    n = len(total_scores)
    g = max(1, int(np.floor(n * top_frac)))
    # Ordenamos por total
    orden = total_scores.sort_values(ascending=False).index
    top_idx = orden[:g]
    bottom_idx = orden[-g:]
    p_top = col_scores.loc[top_idx].mean(skipna=True)
    p_bot = col_scores.loc[bottom_idx].mean(skipna=True)
    # Si ambos son NaN (todos omitidos en ambos grupos), devolvemos NaN
    if pd.isna(p_top) and pd.isna(p_bot):
        return np.nan
    # Reemplazamos NaN por 0 para poder restar coherentemente si un grupo no respondió
    p_top = 0.0 if pd.isna(p_top) else p_top
    p_bot = 0.0 if pd.isna(p_bot) else p_bot
    return p_top - p_bot

def point_biserial_parte_total(col_scores, total_scores):
    # r_pb entre el ítem y el total SIN ese ítem (evita inflación)
    # Si todos correctos o todos incorrectos (varianza 0), corr = NaN
    item = col_scores.fillna(0)  # omitidas como 0 para correlación
    if item.nunique(dropna=False) <= 1:
        return np.nan
    total_sin = (total_scores - item)
    # Si total_sin tiene varianza 0 (poco probable), corr = NaN
    if total_sin.nunique() <= 1:
        return np.nan
    return item.corr(total_sin)

# -------------------------
# 4) Calcular métricas por pregunta
# -------------------------
registros = []
for i in range(num_preg):
    q = f"Q{i+1}"
    col = df_score[q]

    # Dificultad p = proporción de correctas (ignora omitidas)
    p = col.mean(skipna=True)

    # Discriminación D = p_top - p_bottom (27% extremos)
    D = indice_discriminacion(col, total, top_frac=0.27)

    # r_pb (ítem vs total sin ítem)
    rpb = point_biserial_parte_total(col, total)

    # Conteos
    n_omitidas = col.isna().sum()
    n_correctas = int((col == 1).sum())
    n_incorrectas = int((col == 0).sum())

    registros.append({
        "pregunta": i+1,
        "dificultad": None if pd.isna(p) else round(float(p), 3),
        "discriminacion": None if pd.isna(D) else round(float(D), 3),
        "r_pb": None if pd.isna(rpb) else round(float(rpb), 3),
        "n_correctas": n_correctas,
        "n_incorrectas": n_incorrectas,
        "n_omitidas": int(n_omitidas)
    })

# -------------------------
# 5) Exportar a XML
# -------------------------
root_out = ET.Element("analisis_items")
meta = ET.SubElement(root_out, "metadata")
ET.SubElement(meta, "num_preguntas").text = str(num_preg)
ET.SubElement(meta, "num_alumnos").text = str(len(df_score))

for reg in registros:
    item = ET.SubElement(root_out, "item", numero=str(reg["pregunta"]))
    ET.SubElement(item, "dificultad").text = "" if reg["dificultad"] is None else f"{reg['dificultad']:.3f}"
    ET.SubElement(item, "discriminacion").text = "" if reg["discriminacion"] is None else f"{reg['discriminacion']:.3f}"
    ET.SubElement(item, "r_pb").text = "" if reg["r_pb"] is None else f"{reg['r_pb']:.3f}"
    conteos = ET.SubElement(item, "conteos")
    ET.SubElement(conteos, "correctas").text = str(reg["n_correctas"])
    ET.SubElement(conteos, "incorrectas").text = str(reg["n_incorrectas"])
    ET.SubElement(conteos, "omitidas").text = str(reg["n_omitidas"])

tree_out = ET.ElementTree(root_out)
tree_out.write("analisis_items.xml", encoding="utf-8", xml_declaration=True)

print(" Archivo generado: analisis_items.xml")
