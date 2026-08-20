"""Variable respuesta binaria (high_cyano) y control de data leakage.

Ejercicio 2 del Laboratorio 4

Este módulo:
- construye la variable respuesta binaria high_cyano a partir de la columna
  cyano del dataset tabular entregado por Persona 1 (no recalcula cyano ni
  toca los raster ni sentinel.py);
- documenta el criterio científico usado para el punto de corte;
- analiza la distribución de la respuesta globalmente, por lago y por fecha;
- evalúa el balance de clases y sus consecuencias para el entrenamiento;
- identifica las variables que no pueden usarse como predictoras por haber
  intervenido directa o indirectamente en la construcción de la respuesta.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Punto de corte (2.2)
# ---------------------------------------------------------------------------
#
# El índice "cyano" de este proyecto es una estimación de clorofila-a
# (µg/L) obtenida con una adaptación numérica del script oficial
# "Cyanobacteria Chlorophyll-a NDCI L1C" de CyanoLakes (Kravitz & Matthews,
# 2020) -- ver src/indices.py::CYANO_EVALSCRIPT. Por eso el punto de corte
# se justifica con las guías de clorofila-a para cianobacterias en agua
# recreativa de la OMS, la misma referencia que usa CyanoLakes para sus
# propios niveles de alerta:
#
#   World Health Organization (2021). Guidelines on Recreational Water
#   Quality: Volume 1 - Coastal and Fresh Waters. Geneva: WHO.
#   (actualiza WHO 2003 "Guidelines for safe recreational water
#   environments"; ver también Chorus & Testai, "Toxic Cyanobacteria in
#   Water", 2021, capítulo 5).
#
#   Niveles de biomasa de cianobacteria (como clorofila-a, con dominancia
#   de cianobacterias):
#     - Vigilancia:   1  - 12 µg/L
#     - Alerta 1:     12 - 24 µg/L
#     - Alerta 2:     > 24 µg/L (o formación de nata / transparencia baja)
#
#   CyanoLakes (mismos autores del script usado en este proyecto) aplica
#   estos mismos cortes operativamente: <10-12 µg/L = agua mayormente
#   clara/riesgo bajo; 12-24 µg/L = inicio de riesgo moderado-alto
#   (Alerta 1); >24 µg/L = Alerta 2, actividades de contacto total no
#   recomendadas.
#
# Se usa el límite inferior de Alerta 1 (12 µg/L) como punto de corte
# binario: por debajo, el agua está en Vigilancia o mejor (riesgo bajo);
# desde ahí en adelante, la OMS ya considera biomasa de cianobacteria
# suficiente para activar una alerta formal (riesgo moderado a alto).
HIGH_CYANO_THRESHOLD_UGL = 12.0

THRESHOLD_SOURCE = (
    "World Health Organization (2021). Guidelines on Recreational Water "
    "Quality: Volume 1 - Coastal and Fresh Waters. Geneva: WHO "
    "(actualiza WHO 2003/2009). Niveles de clorofila-a con dominancia de "
    "cianobacterias: Vigilancia 1-12 µg/L, Alerta 1 12-24 µg/L, Alerta 2 "
    ">24 µg/L. Ver también Chorus & Testai (eds.), Toxic Cyanobacteria in "
    "Water, 2021, cap. 5; y CyanoLakes (Kravitz & Matthews, 2020), autores "
    "del script de cianobacteria adaptado en este proyecto (src/indices.py), "
    "que aplica estos mismos cortes."
)

RESPONSE_COL = "high_cyano"
CYANO_COL = "cyano"

RESPONSE_SUMMARY_CSV = config.TABLES_DIR / "response_variable_summary.csv"
RESPONSE_BY_DATE_CSV = config.TABLES_DIR / "response_by_date.csv"


# ---------------------------------------------------------------------------
# 2.1 - Construcción de la variable respuesta
# ---------------------------------------------------------------------------

def build_high_cyano(
    df: pd.DataFrame,
    threshold: float = HIGH_CYANO_THRESHOLD_UGL,
    cyano_col: str = CYANO_COL,
) -> pd.DataFrame:
    """Agrega la columna high_cyano (0/1) al dataset tabular de Persona 1.

    0 = ausencia o baja presencia (cyano <= threshold)
    1 = alta presencia (cyano > threshold)

    No modifica el DataFrame recibido -- regresa una copia.
    """
    if cyano_col not in df.columns:
        raise ValueError(f"el dataset no tiene la columna '{cyano_col}'")

    out = df.copy()
    out[RESPONSE_COL] = (out[cyano_col] > threshold).astype(int)
    return out


# ---------------------------------------------------------------------------
# 2.3 - Distribución de la respuesta
# ---------------------------------------------------------------------------

def response_distribution(df: pd.DataFrame) -> dict:
    """Distribución de high_cyano globalmente, por lago y por lago+fecha."""
    if RESPONSE_COL not in df.columns:
        raise ValueError(f"el dataset no tiene la columna '{RESPONSE_COL}'; corre build_high_cyano() primero")

    global_counts = df[RESPONSE_COL].value_counts().sort_index()
    global_pct = (df[RESPONSE_COL].value_counts(normalize=True).sort_index() * 100).round(2)

    by_lake_counts = df.groupby("lake")[RESPONSE_COL].value_counts().unstack(fill_value=0)
    by_lake_pct = (df.groupby("lake")[RESPONSE_COL].value_counts(normalize=True).unstack(fill_value=0) * 100).round(2)

    by_date = (
        df.groupby(["lake", "date"])[RESPONSE_COL]
        .agg(n_observaciones="size", proporcion_alta="mean")
        .reset_index()
    )
    by_date["proporcion_alta"] = (by_date["proporcion_alta"] * 100).round(2)
    by_date = by_date.sort_values(["lake", "date"]).reset_index(drop=True)

    return {
        "global_counts": global_counts,
        "global_pct": global_pct,
        "by_lake_counts": by_lake_counts,
        "by_lake_pct": by_lake_pct,
        "by_date": by_date,
    }


def save_response_distribution_csv(dist: dict) -> tuple[Path, Path]:
    """Guarda un resumen global+por lago y la tabla por lago+fecha."""
    config.ensure_output_dirs()

    summary_rows = [
        {"scope": "global", "lake": "todos", "class_0_pct": dist["global_pct"].get(0, 0.0), "class_1_pct": dist["global_pct"].get(1, 0.0)}
    ]
    for lake in dist["by_lake_pct"].index:
        summary_rows.append({
            "scope": "por_lago",
            "lake": lake,
            "class_0_pct": dist["by_lake_pct"].loc[lake].get(0, 0.0),
            "class_1_pct": dist["by_lake_pct"].loc[lake].get(1, 0.0),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESPONSE_SUMMARY_CSV, index=False)
    print(f"[ok] resumen de distribución de la respuesta guardado en {RESPONSE_SUMMARY_CSV}")

    dist["by_date"].to_csv(RESPONSE_BY_DATE_CSV, index=False)
    print(f"[ok] distribución por fecha guardada en {RESPONSE_BY_DATE_CSV}")

    return RESPONSE_SUMMARY_CSV, RESPONSE_BY_DATE_CSV


def plot_response_distribution(dist: dict) -> dict[str, Path]:
    """Genera las gráficas de distribución de la respuesta: global+por lago
    (barras) y evolución por fecha (líneas, una por lago).
    """
    config.ensure_output_dirs()
    paths = {}

    # --- barras: global y por lago ---
    fig, ax = plt.subplots(figsize=(8, 5))
    lakes = list(dist["by_lake_pct"].index)
    x = np.arange(len(lakes) + 1)
    width = 0.35

    labels = ["Global"] + [lake.capitalize() for lake in lakes]
    class0 = [dist["global_pct"].get(0, 0.0)] + [dist["by_lake_pct"].loc[lake].get(0, 0.0) for lake in lakes]
    class1 = [dist["global_pct"].get(1, 0.0)] + [dist["by_lake_pct"].loc[lake].get(1, 0.0) for lake in lakes]

    ax.bar(x - width / 2, class0, width, label="0 (baja/ausente)", color="#3a7ca5")
    ax.bar(x + width / 2, class1, width, label="1 (alta)", color="#c1440e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% de observaciones")
    ax.set_title(f"Distribución de high_cyano (umbral = {HIGH_CYANO_THRESHOLD_UGL:.0f} µg/L)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    balance_path = config.FIGURES_DIR / "response_balance.png"
    fig.savefig(balance_path, dpi=150)
    plt.close(fig)
    paths["balance"] = balance_path
    print(f"[ok] gráfica de balance de clases guardada en {balance_path}")

    # --- evolución por fecha, una linea por lago ---
    import datetime as dt

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"atitlan": "#2a7f62", "amatitlan": "#b5541c"}
    for lake in lakes:
        lake_rows = dist["by_date"][dist["by_date"]["lake"] == lake]
        dates = [dt.date.fromisoformat(d) for d in lake_rows["date"]]
        ax.plot(dates, lake_rows["proporcion_alta"], marker="o", linewidth=2, color=colors.get(lake), label=lake.capitalize())

    ax.set_title("Proporción de observaciones con high_cyano=1 por fecha")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("% de observaciones con alta presencia")
    fig.autofmt_xdate(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    evolution_path = config.FIGURES_DIR / "response_evolution_by_date.png"
    fig.savefig(evolution_path, dpi=150)
    plt.close(fig)
    paths["evolution"] = evolution_path
    print(f"[ok] gráfica de evolución de la respuesta guardada en {evolution_path}")

    return paths


# ---------------------------------------------------------------------------
# 2.4 - Balance de clases
# ---------------------------------------------------------------------------

def assess_class_balance(dist: dict) -> dict:
    """Evalúa si hay desbalance entre clases a partir de la distribución
    global y describe las consecuencias esperadas para entrenamiento y
    evaluación.
    """
    pct0 = float(dist["global_pct"].get(0, 0.0))
    pct1 = float(dist["global_pct"].get(1, 0.0))
    minority_pct = min(pct0, pct1)
    majority_pct = max(pct0, pct1)
    minority_class = 1 if pct1 <= pct0 else 0
    ratio = majority_pct / minority_pct if minority_pct > 0 else float("inf")

    if minority_pct >= 40:
        severity = "balanceado"
    elif minority_pct >= 20:
        severity = "desbalance leve"
    elif minority_pct >= 10:
        severity = "desbalance moderado"
    else:
        severity = "desbalance severo"

    return {
        "pct_class_0": round(pct0, 2),
        "pct_class_1": round(pct1, 2),
        "minority_class": minority_class,
        "minority_pct": round(minority_pct, 2),
        "ratio_majority_to_minority": round(ratio, 2) if ratio != float("inf") else None,
        "severity": severity,
    }


def build_balance_interpretation(balance: dict) -> str:
    """Redacta la interpretación de 2.3 y 2.4 (distribución + desbalance)."""
    lines = ["## Ejercicio 2: variable respuesta (high_cyano)\n"]
    lines.append(
        f"**Punto de corte**: {HIGH_CYANO_THRESHOLD_UGL:.0f} µg/L de clorofila-a "
        f"(1 = alta presencia, cyano > {HIGH_CYANO_THRESHOLD_UGL:.0f}; 0 = ausente/baja). "
        f"{THRESHOLD_SOURCE}"
    )

    lines.append("\n### 2.3 Distribución de la respuesta\n")
    lines.append(
        f"- Globalmente: {balance['pct_class_0']:.1f}% de las observaciones caen en clase 0 "
        f"(baja/ausente) y {balance['pct_class_1']:.1f}% en clase 1 (alta presencia). "
        f"Ver `response_balance.png` (por lago) y `response_evolution_by_date.png` "
        f"(evolución por fecha) para el detalle."
    )

    lines.append("\n### 2.4 Balance de clases\n")
    lines.append(
        f"- La clase minoritaria es {balance['minority_class']} con "
        f"{balance['minority_pct']:.1f}% de las observaciones "
        f"(razón mayoría:minoría ≈ {balance['ratio_majority_to_minority']}:1). "
        f"Clasificación: **{balance['severity']}**."
    )
    if balance["severity"] in ("desbalance moderado", "desbalance severo"):
        lines.append(
            "- Consecuencias esperadas: un modelo entrenado sin corrección puede sesgarse hacia "
            "la clase mayoritaria y lograr un accuracy alto simplemente prediciendo la clase "
            "dominante, sin detectar bien los casos de alta presencia de cianobacteria (que son "
            "justamente los que importan desde el punto de vista de salud pública). Accuracy por "
            "sí sola sería una métrica engañosa; conviene priorizar recall/F1/ROC-AUC de la clase "
            "1, y considerar balanceo (class_weight, sobremuestreo/submuestreo, o umbrales de "
            "decisión ajustados) al entrenar los modelos en el ejercicio 4."
        )
    else:
        lines.append(
            "- Con este nivel de balance, accuracy es una métrica razonable como referencia "
            "inicial, aunque sigue siendo buena práctica reportar también precision/recall/F1 "
            "por clase."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2.5 - Control de leakage
# ---------------------------------------------------------------------------

def leakage_report() -> dict:
    """Identifica qué columnas del dataset tabular (contrato de Persona 1:
    lake,date,longitude,latitude,b2,b3,b4,b8,ndvi,ndwi,cyano) no pueden
    usarse como predictoras por haber intervenido en la construcción de
    high_cyano, directa o indirectamente.

    Basado en cómo se calcula cyano en src/indices.py::CYANO_EVALSCRIPT:
    - el valor numérico de clorofila-a sale de un NDCI que usa B04 y B05
      (chl = f((B05-B04)/(B05+B04)));
    - la máscara de agua que decide qué píxeles cuentan usa B02,B03,B04,
      B08,B11,B12 (r,g,b,nir,swir1,swir2).
    - el dataset tabular de Persona 1 (tabular.py) solo incluye b2,b3,b4,b8
      (no incluye B05, B07, B8A, B11 ni B12).
    """
    return {
        "prohibited": {
            "cyano": (
                "high_cyano se construye directamente como cyano > umbral. Es una "
                "transformación 1 a 1 de la respuesta -- incluirla como predictor "
                "sería leakage total y garantizaría una separación perfecta artificial."
            ),
        },
        "caution": {
            "b4": (
                "B04 es uno de los dos insumos directos del NDCI que produce el valor "
                "numérico de cyano (chl = f((B05-B04)/(B05+B04))). El otro insumo, B05, "
                "no está en el dataset tabular, así que b4 por sí solo NO determina cyano "
                "-- pero sí participa matemáticamente en su cálculo. Se recomienda "
                "evaluarlo con cautela (por ejemplo comparando desempeño del modelo con y "
                "sin b4) antes de darlo por seguro como predictor. También participa en la "
                "máscara de agua (water body classifier) que decide qué píxeles se incluyen "
                "en el dataset, junto con b2, b3 y b8."
            ),
        },
        "safe": {
            "b2": "No participa en el NDCI de cianobacteria; solo interviene (junto con b3,b4,b8) en la máscara de agua que decide inclusión/exclusión de píxeles, no en el valor numérico de cyano.",
            "b3": "No participa en el NDCI de cianobacteria. Se usa para NDWI y en la máscara de agua.",
            "b8": "No participa en el NDCI de cianobacteria (usa B04 y B05, no B08). Se usa para NDVI/NDWI y en la máscara de agua.",
            "ndvi": "Calculado a partir de b4 y b8; no involucra B05, así que no reproduce el NDCI de cyano.",
            "ndwi": "Calculado a partir de b3 y b8; no interviene en el cálculo de cyano.",
            "longitude": "Coordenada geográfica, no interviene en el cálculo espectral de cyano.",
            "latitude": "Coordenada geográfica, no interviene en el cálculo espectral de cyano.",
        },
    }


def build_leakage_interpretation(report: dict) -> str:
    """Redacta la sección 2.5 del informe (variables prohibidas por leakage)."""
    lines = ["\n### 2.5 Variables excluidas por data leakage\n"]
    lines.append("**Prohibidas (excluir siempre de X):**")
    for var, reason in report["prohibited"].items():
        lines.append(f"- `{var}`: {reason}")

    lines.append("\n**Uso con precaución (coordinar con Persona 3):**")
    for var, reason in report["caution"].items():
        lines.append(f"- `{var}`: {reason}")

    lines.append("\n**Seguras como predictoras (no participan en el cálculo de cyano):**")
    for var, reason in report["safe"].items():
        lines.append(f"- `{var}`: {reason}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run_response_variable_analysis(df: pd.DataFrame) -> dict:
    """Ejecuta el ejercicio 2 completo (2.1-2.5) sobre el dataset tabular de
    Persona 1 y regresa el dataset con high_cyano ya agregado más todos los
    resultados intermedios.
    """
    config.ensure_output_dirs()

    df_with_response = build_high_cyano(df)

    dist = response_distribution(df_with_response)
    save_response_distribution_csv(dist)
    figure_paths = plot_response_distribution(dist)

    balance = assess_class_balance(dist)
    balance_interpretation = build_balance_interpretation(balance)

    leakage = leakage_report()
    leakage_interpretation = build_leakage_interpretation(leakage)

    full_interpretation = balance_interpretation + "\n" + leakage_interpretation
    print("\n" + full_interpretation)

    return {
        "df": df_with_response,
        "distribution": dist,
        "balance": balance,
        "leakage": leakage,
        "figure_paths": figure_paths,
        "interpretation": full_interpretation,
    }


if __name__ == "__main__":
    # requiere que Persona 1 ya haya corrido generate_tabular_dataset()
    from src.tabular import generate_tabular_dataset

    df = generate_tabular_dataset()
    run_response_variable_analysis(df)