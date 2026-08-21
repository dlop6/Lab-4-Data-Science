"""EDA y selección de predictores (Parte 2, incisos 1.5, 1.6 y 3.1-3.3).

Este módulo:
- carga el dataset final (22 CSV de Persona 1 en src/tabular.py + high_cyano de
  Persona 2 en src/response_variable.py), sin reimplementar ninguna de las dos
  cosas;
- caracteriza las variables candidatas con estadísticas y visualizaciones (1.5);
- documenta la limpieza ya aplicada aguas arriba, sin repetirla (1.6);
- aplica (no redefine) la clasificación de leakage de Persona 2 para decidir qué
  columnas pueden evaluarse como predictoras;
- construye el catálogo de predictores candidatos con su justificación individual,
  detecta pares muy correlacionados entre sí, y cierra con la lista definitiva de
  predictores (3.1-3.3).

Decisión de alcance (no un olvido): `b4` se excluye de los predictores finales por
precaución -- participa matemáticamente en el NDCI de cianobacteria aunque no lo
determina por sí solo (ver src/response_variable.py::leakage_report(), categoría
"caution"). `longitude`/`latitude` sí se incluyen como candidatas.

No se agrega scikit-learn, scipy ni seaborn: nada en la división de tareas de esta
parte exige VIF formal de multicolinealidad ni un método de correlación distinto a
Pearson (que el resto del repo ya usa en src/correlation.py), así que no se agrega
una dependencia sin necesidad documentada (KISS/YAGNI).

Nota DRY reconocida: src/tabular.py define su propia constante local TABULAR_DIR
(no se toca ese archivo, está fuera del alcance de Persona 3); este módulo usa la
misma ruta pero resuelta desde config.TABULAR_DIR, la fuente de verdad única.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.response_variable import HIGH_CYANO_THRESHOLD_UGL, build_high_cyano, leakage_report

RESPONSE_COL = "high_cyano"

# variables candidatas: todas las columnas numericas del dataset final salvo
# lake/date (identificadores, no candidatas) y high_cyano (es la respuesta).
CANDIDATE_COLUMNS = ("b2", "b3", "b4", "b8", "ndvi", "ndwi", "cyano", "longitude", "latitude")

# muestreo solo para graficas (las estadisticas se calculan sobre el dataset
# completo). mismo patron ya usado en src/distributions.py y src/correlation.py.
MAX_PLOT_POINTS = 100_000
RANDOM_SEED = 42

# umbral para senalar pares de predictores muy correlacionados entre si (3.2).
# valor de referencia comun en la practica, no requiere ninguna libreria nueva.
HIGH_CORRELATION_THRESHOLD = 0.9
# umbral para decidir exclusion automatica por redundancia casi total (3.3).
REDUNDANCY_THRESHOLD = 0.95

PREDICTORS_FIGURES_DIR = config.FIGURES_DIR / "predictors"

DESCRIPTIVE_STATS_CSV = config.TABLES_DIR / "predictors_descriptive_stats.csv"
STATS_BY_CLASS_CSV = config.TABLES_DIR / "predictors_stats_by_class.csv"
FINAL_PREDICTORS_CSV = config.TABLES_DIR / "final_predictors.csv"


# ---------------------------------------------------------------------------
# Fase 0 - carga del dataset final
# ---------------------------------------------------------------------------

def load_final_dataset() -> pd.DataFrame:
    """Lee los 22 CSV de src/tabular.py y agrega high_cyano (src/response_variable.py).

    No descarga nada de red ni toca raster -- solo lee los CSV ya generados.
    Falla explicitamente (FileNotFoundError con lago/fecha/ruta) si algun CSV
    no existe, en vez de producir un dataset incompleto en silencio.
    """
    dataframes = []
    for lake in config.LAKES:
        for date in config.DATES[lake]:
            csv_path = config.TABULAR_DIR / lake / f"{date}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"no existe el csv tabular de {lake} {date}: {csv_path}. "
                    f"corre generate_tabular_dataset() (src/tabular.py) primero."
                )
            dataframes.append(pd.read_csv(csv_path))

    df = pd.concat(dataframes, ignore_index=True)
    df = build_high_cyano(df)

    print(f"[ok] dataset final cargado: {len(df):,} filas, {len(df.columns)} columnas")
    return df


# ---------------------------------------------------------------------------
# Fase 1 - EDA de variables candidatas (1.5, 1.6)
# ---------------------------------------------------------------------------

def compute_descriptive_stats(df: pd.DataFrame, columns: tuple[str, ...] = CANDIDATE_COLUMNS) -> pd.DataFrame:
    """Estadisticas descriptivas por variable candidata: count, mean, std, min,
    25/50/75%, max y porcentaje de valores faltantes (calculado, no asumido).
    """
    rows = []
    for col in columns:
        desc = df[col].describe()
        rows.append({
            "variable": col,
            "count": desc["count"],
            "mean": desc["mean"],
            "std": desc["std"],
            "min": desc["min"],
            "25%": desc["25%"],
            "50%": desc["50%"],
            "75%": desc["75%"],
            "max": desc["max"],
            "missing_pct": float(df[col].isna().mean() * 100),
        })
    return pd.DataFrame(rows)


def save_descriptive_stats_csv(stats: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/predictors_descriptive_stats.csv."""
    config.ensure_output_dirs()
    output_path = output_path or DESCRIPTIVE_STATS_CSV
    stats.to_csv(output_path, index=False)
    print(f"[ok] estadisticas descriptivas guardadas en {output_path}")
    return output_path


def compute_stats_by_class(df: pd.DataFrame, columns: tuple[str, ...] = CANDIDATE_COLUMNS) -> pd.DataFrame:
    """Media y desviacion de cada variable candidata, agrupada por high_cyano
    (0 vs 1), mas la diferencia de medias entre clases.
    """
    rows = []
    for col in columns:
        by_class = df.groupby(RESPONSE_COL)[col].agg(["mean", "std"])
        mean_class0 = float(by_class.loc[0, "mean"])
        mean_class1 = float(by_class.loc[1, "mean"])
        rows.append({
            "variable": col,
            "mean_class0": mean_class0,
            "mean_class1": mean_class1,
            "std_class0": float(by_class.loc[0, "std"]),
            "std_class1": float(by_class.loc[1, "std"]),
            "diff_means": mean_class1 - mean_class0,
        })
    return pd.DataFrame(rows)


def save_stats_by_class_csv(stats: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/predictors_stats_by_class.csv."""
    config.ensure_output_dirs()
    output_path = output_path or STATS_BY_CLASS_CSV
    stats.to_csv(output_path, index=False)
    print(f"[ok] estadisticas por clase guardadas en {output_path}")
    return output_path


def _sample_for_plot(series: pd.Series, max_points: int = MAX_PLOT_POINTS, seed: int = RANDOM_SEED) -> pd.Series:
    """Muestra reproducible solo para graficar (no afecta ningun calculo
    estadistico, que siempre se hace sobre el dataset completo).
    """
    if len(series) <= max_points:
        return series
    return series.sample(n=max_points, random_state=seed)


def plot_boxplots_by_class(df: pd.DataFrame, columns: tuple[str, ...] = CANDIDATE_COLUMNS) -> list[Path]:
    """Un boxplot por variable candidata, comparando su distribucion en
    high_cyano=0 vs high_cyano=1.
    """
    config.ensure_output_dirs()
    PREDICTORS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for col in columns:
        class0 = _sample_for_plot(df.loc[df[RESPONSE_COL] == 0, col])
        class1 = _sample_for_plot(df.loc[df[RESPONSE_COL] == 1, col])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.boxplot([class0, class1], tick_labels=["0 (baja/ausente)", "1 (alta)"], showfliers=False)
        ax.set_title(f"Distribución de {col} por clase (high_cyano)")
        ax.set_ylabel(col)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        output_path = PREDICTORS_FIGURES_DIR / f"boxplot_{col}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        paths.append(output_path)
        print(f"[ok] boxplot de {col} guardado en {output_path}")

    return paths


def plot_correlation_heatmap(df: pd.DataFrame, columns: tuple[str, ...] = CANDIDATE_COLUMNS) -> tuple[Path, pd.DataFrame]:
    """Matriz de correlacion de Pearson entre todas las variables candidatas
    (incluye cyano aqui, util para ver que tanto correlaciona cada candidata
    con cyano; la exclusion de cyano como predictor se aplica despues, en la
    seleccion final, no en este EDA exploratorio).
    """
    config.ensure_output_dirs()
    PREDICTORS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corr = df[list(columns)].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlación de Pearson entre variables candidatas")
    fig.colorbar(im, ax=ax, label="Coeficiente de correlación")
    fig.tight_layout()

    output_path = PREDICTORS_FIGURES_DIR / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ok] heatmap de correlación guardado en {output_path}")

    return output_path, corr


def build_data_preparation_notes() -> str:
    """Documenta la limpieza ya aplicada por Persona 1 (src/tabular.py), sin
    repetirla ni contradecirla. Solo documenta lo verificable en ese código.
    """
    lines = ["## Preparación y limpieza del dataset (aplicada aguas arriba)\n"]
    lines.append(
        "Persona 1 (src/tabular.py::raster_to_dataframe) ya aplicó, antes de que "
        "este dataset llegara a Persona 3, las siguientes reglas de limpieza -- "
        "no se repiten ni se reimplementan aquí:"
    )
    lines.append("- Máscara del contorno real del lago (GeoJSON), excluyendo tierra fuera del lago.")
    lines.append(
        "- Exclusión de píxeles con clases problemáticas de la Scene Classification "
        "Layer (SCL): NoData, saturados/defectuosos, sombra de nube, nube media/alta "
        "probabilidad, cirros delgados y nieve/hielo."
    )
    lines.append("- Exclusión de píxeles con dataMask=0 (sin dato válido de Sentinel-2).")
    lines.append(
        "- Eliminación de observaciones donde NDVI o NDWI no pudieron calcularse por "
        "denominador cero."
    )
    lines.append(
        "\nEsto ya se refleja en el 0% de valores faltantes observado en las "
        "estadísticas descriptivas de este módulo: no significa que no hiciera falta "
        "limpieza, sino que esa limpieza ya se hizo antes de generar los CSV."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fase 2 - control de leakage (aplicado, no redefinido)
# ---------------------------------------------------------------------------

def apply_leakage_exclusions(candidates: tuple[str, ...] = CANDIDATE_COLUMNS) -> dict:
    """Aplica la clasificacion de leakage ya documentada por Persona 2
    (src/response_variable.py::leakage_report()) a la lista de candidatas.

    No redefine el leakage -- solo separa `candidates` segun lo que Persona 2
    ya clasifico como prohibido/con cautela/seguro. Si esa clasificacion
    cambia en el futuro, esta funcion refleja el cambio automaticamente (no
    hay ninguna lista de nombres hardcodeada aca).
    """
    report = leakage_report()

    excluded_prohibited = [c for c in candidates if c in report["prohibited"]]
    excluded_caution = [c for c in candidates if c in report["caution"]]
    retained = [c for c in candidates if c not in report["prohibited"] and c not in report["caution"]]

    return {
        "excluded_prohibited": excluded_prohibited,
        "excluded_caution": excluded_caution,
        "retained": retained,
        "reasons": {
            **{c: report["prohibited"][c] for c in excluded_prohibited},
            **{c: report["caution"][c] for c in excluded_caution},
        },
    }


# ---------------------------------------------------------------------------
# Fase 3 - definicion formal de predictores (3.1-3.3)
# ---------------------------------------------------------------------------

# catalogo de justificaciones, una entrada especifica por variable (no se
# genera texto generico/copia-pega entre variables).
_PREDICTOR_DESCRIPTIONS = {
    "b2": {
        "type": "banda_espectral",
        "description": "Reflectancia en la banda azul (B02) de Sentinel-2.",
        "rationale": (
            "Interviene junto con b3, b4 y b8 en el clasificador de cuerpo de agua "
            "que usa el propio script de cianobacteria del proyecto; ayuda a "
            "distinguir agua de otras superficies, relevante como contexto espectral."
        ),
    },
    "b3": {
        "type": "banda_espectral",
        "description": "Reflectancia en la banda verde (B03) de Sentinel-2.",
        "rationale": (
            "Es uno de los dos insumos de NDWI y participa en la máscara de agua; "
            "el verde es sensible a pigmentos fotosintéticos suspendidos en el agua, "
            "lo que la vincula plausiblemente con biomasa algal."
        ),
    },
    "b8": {
        "type": "banda_espectral",
        "description": "Reflectancia en el infrarrojo cercano (B08) de Sentinel-2.",
        "rationale": (
            "Insumo de NDVI y NDWI, y parte de la máscara de agua; el infrarrojo "
            "cercano distingue fuertemente agua de vegetación/tierra, útil como "
            "contexto para interpretar el resto de índices."
        ),
    },
    "ndvi": {
        "type": "indice_espectral",
        "description": "Índice de vegetación normalizado, calculado a partir de b4 y b8.",
        "rationale": (
            "Puede detectar materia vegetal/orgánica visible en superficie (ej. "
            "acumulaciones de algas o plantas acuáticas flotantes), un contexto "
            "plausible para floraciones algales."
        ),
    },
    "ndwi": {
        "type": "indice_espectral",
        "description": "Índice de agua normalizado, calculado a partir de b3 y b8.",
        "rationale": (
            "Distingue agua clara de zonas con más turbidez o materia en suspensión; "
            "un NDWI más bajo dentro del cuerpo de agua puede coincidir con zonas de "
            "mayor concentración de partículas, incluida biomasa de cianobacteria."
        ),
    },
    "longitude": {
        "type": "caracteristica_espacial",
        "description": "Coordenada geográfica (longitud) del centro del píxel.",
        "rationale": (
            "No interviene en el cálculo espectral de cyano (segura contra leakage), "
            "pero es una característica de ubicación, no espectral: no generaliza "
            "entre lagos ni necesariamente a fechas futuras -- riesgo de que el "
            "modelo memorice ubicaciones específicas en vez de aprender un patrón "
            "espectral transferible. Se incluye como candidata con esta limitación "
            "documentada explícitamente."
        ),
    },
    "latitude": {
        "type": "caracteristica_espacial",
        "description": "Coordenada geográfica (latitud) del centro del píxel.",
        "rationale": (
            "Misma consideración que longitude: segura contra leakage, pero con el "
            "mismo riesgo de sobreajuste geográfico en vez de señal espectral "
            "generalizable."
        ),
    },
}


def build_predictor_catalog(retained: list[str]) -> list[dict]:
    """Catálogo de predictores candidatos retenidos (tras leakage), con tipo,
    descripción y justificación individual -- lo que pide explícitamente el
    PDF de división de tareas para cada predictor.
    """
    catalog = []
    for name in retained:
        info = _PREDICTOR_DESCRIPTIONS[name]
        catalog.append({
            "name": name,
            "type": info["type"],
            "description": info["description"],
            "rationale": info["rationale"],
        })
    return catalog


def flag_highly_correlated_pairs(
    corr_matrix: pd.DataFrame,
    columns: list[str],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Señala pares de variables (dentro de `columns`) con |correlación| >=
    threshold en `corr_matrix` (ya calculada en la Fase 1, no se recalcula).

    No excluye nada por sí sola -- solo reporta los pares para que la
    interpretación final decida.
    """
    pairs = []
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            value = float(corr_matrix.loc[col_a, col_b])
            if abs(value) >= threshold:
                pairs.append((col_a, col_b, value))
    return pairs


def build_final_predictor_list(
    catalog: list[dict],
    correlated_pairs: list[tuple[str, str, float]],
) -> tuple[list[str], list[dict]]:
    """Lista definitiva de predictores: por defecto todo el catálogo, salvo
    redundancia casi total (|r| >= REDUNDANCY_THRESHOLD) entre un par, donde se
    prioriza mantener el índice espectral (ndvi/ndwi) sobre la banda cruda que
    resume, porque el índice ya combina la señal de ambas bandas. Esta regla
    de desempate es explícita, no aplicada en silencio.

    Retorna (lista_final, exclusiones) donde cada exclusión documenta motivo.
    """
    names = [entry["name"] for entry in catalog]
    excluded_by_redundancy: list[dict] = []
    excluded_names: set[str] = set()

    for col_a, col_b, value in correlated_pairs:
        if abs(value) < REDUNDANCY_THRESHOLD:
            continue
        if col_a in excluded_names or col_b in excluded_names:
            continue

        type_a = next(e["type"] for e in catalog if e["name"] == col_a)
        type_b = next(e["type"] for e in catalog if e["name"] == col_b)

        if type_a == "indice_espectral" and type_b == "banda_espectral":
            drop = col_b
        elif type_b == "indice_espectral" and type_a == "banda_espectral":
            drop = col_a
        else:
            # mismo tipo en ambos lados: se conserva el primero en orden
            # alfabetico para que la regla sea determinista y reproducible.
            drop = max(col_a, col_b)

        excluded_names.add(drop)
        excluded_by_redundancy.append({
            "name": drop,
            "reason": (
                f"redundante con '{col_a if drop == col_b else col_b}' "
                f"(correlación de Pearson = {value:.3f}, >= umbral de "
                f"{REDUNDANCY_THRESHOLD})."
            ),
        })

    final_list = [name for name in names if name not in excluded_names]
    return final_list, excluded_by_redundancy


def save_final_predictor_list_csv(final_list: list[str], output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/final_predictors.csv (una columna: predictor)."""
    config.ensure_output_dirs()
    output_path = output_path or FINAL_PREDICTORS_CSV
    pd.DataFrame({"predictor": final_list}).to_csv(output_path, index=False)
    print(f"[ok] lista definitiva de predictores guardada en {output_path}")
    return output_path


def build_predictor_selection_interpretation(
    leakage: dict,
    catalog: list[dict],
    correlated_pairs: list[tuple[str, str, float]],
    final_list: list[str],
    redundancy_exclusions: list[dict],
) -> str:
    """Redacta la interpretación final de selección de predictores, separando
    con claridad qué es un hecho observado en los datos de qué es una decisión
    de diseño.
    """
    lines = ["## Selección de predictores (incisos 3.1-3.3)\n"]
    lines.append(
        f"Recordatorio (definido por Persona 2, no redefinido aquí): high_cyano = 1 "
        f"cuando cyano > {HIGH_CYANO_THRESHOLD_UGL:.0f} µg/L de clorofila-a; 0 en caso "
        f"contrario."
    )

    lines.append("\n### Exclusiones por data leakage (definidas por Persona 2, aplicadas aquí)\n")
    for name in leakage["excluded_prohibited"]:
        lines.append(f"- `{name}` (prohibida): {leakage['reasons'][name]}")
    for name in leakage["excluded_caution"]:
        lines.append(f"- `{name}` (excluida por precaución): {leakage['reasons'][name]}")

    lines.append("\n### Catálogo de predictores candidatos retenidos\n")
    for entry in catalog:
        lines.append(f"- `{entry['name']}` ({entry['type']}): {entry['description']} {entry['rationale']}")

    lines.append("\n### Pares altamente correlacionados (observado en los datos)\n")
    if correlated_pairs:
        for col_a, col_b, value in correlated_pairs:
            lines.append(f"- `{col_a}` y `{col_b}`: r = {value:.3f}")
    else:
        lines.append(f"- Ningún par de predictores candidatos superó el umbral de {HIGH_CORRELATION_THRESHOLD}.")

    lines.append("\n### Exclusiones por redundancia (decisión de diseño)\n")
    if redundancy_exclusions:
        for exclusion in redundancy_exclusions:
            lines.append(f"- `{exclusion['name']}`: {exclusion['reason']}")
    else:
        lines.append(
            f"- Ninguna: no hubo pares con |r| >= {REDUNDANCY_THRESHOLD}, así que no se "
            f"forzó ninguna reducción adicional de la lista de candidatas retenidas."
        )

    lines.append(f"\n### Lista definitiva de predictores\n")
    lines.append(", ".join(f"`{name}`" for name in final_list))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fase 4 - orquestador unico
# ---------------------------------------------------------------------------

def run_predictor_analysis(df: pd.DataFrame | None = None) -> dict:
    """Ejecuta el EDA y la selección de predictores completos (1.5, 1.6, 3.1-3.3).

    Si no se recibe df, lo carga con load_final_dataset() (mismo patrón que
    run_exploratory_analysis(temporal_records=None) en src/exploratory.py).
    """
    config.ensure_output_dirs()

    if df is None:
        df = load_final_dataset()

    descriptive_stats = compute_descriptive_stats(df)
    save_descriptive_stats_csv(descriptive_stats)

    stats_by_class = compute_stats_by_class(df)
    save_stats_by_class_csv(stats_by_class)

    boxplot_paths = plot_boxplots_by_class(df)
    heatmap_path, corr_matrix = plot_correlation_heatmap(df)

    preparation_notes = build_data_preparation_notes()
    print("\n" + preparation_notes)

    leakage = apply_leakage_exclusions()
    catalog = build_predictor_catalog(leakage["retained"])
    correlated_pairs = flag_highly_correlated_pairs(corr_matrix, leakage["retained"])
    final_list, redundancy_exclusions = build_final_predictor_list(catalog, correlated_pairs)
    save_final_predictor_list_csv(final_list)

    interpretation = build_predictor_selection_interpretation(
        leakage, catalog, correlated_pairs, final_list, redundancy_exclusions
    )
    print("\n" + interpretation)

    return {
        "descriptive_stats": descriptive_stats,
        "stats_by_class": stats_by_class,
        "figure_paths": {"boxplots": boxplot_paths, "heatmap": heatmap_path},
        "preparation_notes": preparation_notes,
        "leakage_exclusions": leakage,
        "predictor_catalog": catalog,
        "correlated_pairs": correlated_pairs,
        "final_predictors": final_list,
        "redundancy_exclusions": redundancy_exclusions,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    run_predictor_analysis()
