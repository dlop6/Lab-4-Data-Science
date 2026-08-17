"""Análisis exploratorio adicional (ejercicio 8: 8.1, 8.2, 8.4 y 8.5).

Este módulo:
- reutiliza la máscara de polígono de src.spatial.load_masked_raster() para
  quedarse solo con píxeles reales del lago (no reimplementa el enmascarado,
  ni toca sentinel ni las fórmulas espectrales);
- define un único criterio reproducible de "cianobacteria alta": percentil
  75 sobre TODOS los valores válidos de cianobacteria del conjunto completo
  (ambos lagos, las 22 fechas oficiales, ya enmascarados con el geojson de
  cada lago);
- calcula, por lago y fecha, el porcentaje de área del lago sobre ese umbral
  (8.1: high_cyano_percentage);
- calcula persistencia espacial por píxel: en cuántas de las fechas
  oficiales de su lago cada píxel superó el umbral (8.2);
- agrupa las fechas oficiales en época seca/lluviosa para una exploración
  descriptiva de estacionalidad (8.4), dejando explícito que las fechas
  disponibles son pocas y no están distribuidas uniformemente;
- todo el análisis es descriptivo -- no se ajusta ningún modelo estadístico.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

import config
from src.spatial import load_masked_raster

CYANO_FILENAME = "cyano.tif"

HIGH_CYANO_PERCENTAGE_CSV = config.TABLES_DIR / "high_cyano_percentage.csv"

# Umbral de "cianobacteria alta": percentil 75 sobre TODOS los valores
# válidos de cyano.tif del conjunto completo (ambos lagos, 22 fechas, ya
# enmascarados con el geojson de cada lago). Un único número global para
# poder comparar ambos lagos con el mismo criterio -- documentado aquí y
# expuesto en el resultado de run_exploratory_analysis().
HIGH_CYANO_PERCENTILE = 75

# Meses de temporada seca / lluviosa en Guatemala (referencia climática
# general del país -- no un cálculo derivado de los datos de este
# laboratorio).
DRY_SEASON_MONTHS = {11, 12, 1, 2, 3, 4}
RAINY_SEASON_MONTHS = {5, 6, 7, 8, 9, 10}

# Umbral de persistencia "alta" solo para la interpretación (8.2): un pixel
# se reporta como persistente si superó el umbral en más de la mitad de las
# fechas oficiales de su lago. No filtra ni descarta nada.
PERSISTENT_MAJORITY_FRACTION = 0.5


def season_of(date_str: str) -> str:
    """Clasifica una fecha YYYY-MM-DD en 'seca' o 'lluviosa' según el mes."""
    month = int(date_str.split("-")[1])
    return "seca" if month in DRY_SEASON_MONTHS else "lluviosa"


def _load_all_cyano_values(lake: str, date: str) -> np.ndarray:
    """Valores válidos (dentro del polígono del lago) de cyano.tif de una
    fecha. Regresa un array vacío si el raster todavía no existe.
    """
    path = config.raster_dir(lake, date) / CYANO_FILENAME
    if not path.exists():
        return np.array([], dtype="float64")
    array, _ = load_masked_raster(lake, path)
    return array[np.isfinite(array)]


def compute_global_threshold() -> float:
    """Calcula el umbral global de 'cianobacteria alta' (percentil 75) sobre
    todos los valores válidos de cyano.tif de ambos lagos y las 22 fechas
    oficiales.
    """
    all_values = []
    for lake in config.LAKES:
        for date in config.DATES[lake]:
            values = _load_all_cyano_values(lake, date)
            if values.size > 0:
                all_values.append(values)

    if not all_values:
        raise ValueError("no hay ningún cyano.tif válido en todo el proyecto todavía")

    combined = np.concatenate(all_values)
    threshold = float(np.percentile(combined, HIGH_CYANO_PERCENTILE))
    print(
        f"[info] umbral global de cianobacteria alta (percentil {HIGH_CYANO_PERCENTILE}): "
        f"{threshold:.4f} (calculado sobre {combined.size:,} píxeles válidos de ambos lagos)"
    )
    return threshold


def compute_high_cyano_percentage(threshold: float) -> list[dict]:
    """Por lago y fecha: porcentaje de píxeles válidos del lago que superan
    el umbral global (8.1).
    """
    rows = []
    for lake in config.LAKES:
        for date in config.DATES[lake]:
            values = _load_all_cyano_values(lake, date)
            if values.size == 0:
                print(f"[warn] {lake} {date}: sin píxeles válidos, se omite")
                rows.append({"lake": lake, "date": date, "high_cyano_percentage": None})
                continue
            pct = float((values > threshold).sum() / values.size * 100)
            rows.append({"lake": lake, "date": date, "high_cyano_percentage": pct})

    rows.sort(key=lambda r: (r["lake"], r["date"]))
    return rows


def save_high_cyano_percentage_csv(rows: list[dict], output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/high_cyano_percentage.csv con columnas
    lake,date,high_cyano_percentage.
    """
    config.ensure_output_dirs()
    output_path = output_path or HIGH_CYANO_PERCENTAGE_CSV

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lake", "date", "high_cyano_percentage"])
        for r in rows:
            pct = "" if r["high_cyano_percentage"] is None else f"{r['high_cyano_percentage']:.4f}"
            writer.writerow([r["lake"], r["date"], pct])

    print(f"[ok] tabla de porcentaje de área afectada guardada en {output_path}")
    return output_path


def plot_affected_area_evolution(rows: list[dict], output_path: Path | None = None) -> Path:
    """Grafica la evolución del % de área con cianobacteria alta para ambos
    lagos, sobre una línea de tiempo real compartida (8.1).
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.FIGURES_DIR / "affected_area.png")
    colors = {"atitlan": "#2a7f62", "amatitlan": "#b5541c"}

    fig, ax = plt.subplots(figsize=(11, 5))
    for lake in config.LAKES:
        lake_rows = [r for r in rows if r["lake"] == lake and r["high_cyano_percentage"] is not None]
        dates = [dt.date.fromisoformat(r["date"]) for r in lake_rows]
        values = [r["high_cyano_percentage"] for r in lake_rows]
        ax.plot(dates, values, marker="o", linewidth=2, color=colors.get(lake), label=lake.capitalize())

    ax.set_title("Porcentaje del lago con cianobacteria alta a lo largo del tiempo")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Área afectada (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] gráfica de extensión espacial guardada en {output_path}")
    return output_path


def compute_pixel_persistence(lake: str, threshold: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Para un lago: por píxel, cuenta en cuántas de sus fechas oficiales el
    valor de cianobacteria superó el umbral global (8.2).

    También regresa valid_counts (en cuántas fechas ese píxel tuvo dato
    válido) para poder distinguir "fuera del lago / sin datos" de "dentro
    del lago pero nunca sobre el umbral" al graficar e interpretar.
    """
    persistence = None
    valid_counts = None
    meta = None
    n_dates_used = 0

    for date in config.DATES[lake]:
        path = config.raster_dir(lake, date) / CYANO_FILENAME
        if not path.exists():
            continue
        array, m = load_masked_raster(lake, path)

        if persistence is None:
            persistence = np.zeros(array.shape, dtype="int32")
            valid_counts = np.zeros(array.shape, dtype="int32")
            meta = m
        elif array.shape != persistence.shape:
            print(
                f"[warn] {lake} {date}: shape {array.shape} no coincide con "
                f"{persistence.shape} de fechas anteriores, se omite esta fecha "
                f"del cálculo de persistencia"
            )
            continue

        finite = np.isfinite(array)
        valid_counts[finite] += 1
        persistence[finite & (array > threshold)] += 1
        n_dates_used += 1

    if persistence is None:
        raise ValueError(f"no hay ningún cyano.tif válido para {lake} todavía")

    print(f"[info] persistencia de {lake} calculada sobre {n_dates_used} fecha(s)")
    return persistence, valid_counts, meta


def plot_persistence_map(
    lake: str, persistence: np.ndarray, valid_counts: np.ndarray, meta: dict, output_path: Path
) -> Path:
    """Genera el mapa de persistencia espacial de un lago (8.2). Los píxeles
    que nunca tuvieron dato válido (fuera del polígono, o siempre nublados)
    se dejan transparentes para no confundirlos con "válido pero nunca alto".
    """
    bounds = meta["bounds"]
    display = persistence.astype("float64")
    display[valid_counts == 0] = np.nan

    max_dates = len(config.DATES[lake])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        display, cmap="inferno", vmin=0, vmax=max_dates,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    )
    ax.set_title(
        f"Persistencia espacial de cianobacteria alta — {lake.capitalize()}\n"
        f"(número de fechas, de {max_dates}, con valor sobre el umbral)"
    )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.colorbar(im, ax=ax, label="Fechas con cianobacteria alta")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] mapa de persistencia guardado en {output_path}")
    return output_path


def summarize_persistence(lake: str, persistence: np.ndarray, valid_counts: np.ndarray) -> dict:
    """Estadísticas simples de persistencia para la interpretación (8.2):
    qué porcentaje del área del lago (píxeles con al menos un dato válido)
    superó el umbral en la mayoría de sus fechas oficiales.
    """
    ever_valid = valid_counts > 0
    total_pixels = int(ever_valid.sum())
    max_dates = len(config.DATES[lake])
    majority_threshold = max_dates * PERSISTENT_MAJORITY_FRACTION

    never = int(((persistence == 0) & ever_valid).sum())
    persistent = int((persistence >= majority_threshold).sum())
    max_persistence = int(persistence.max()) if persistence.size else 0

    return {
        "lake": lake,
        "total_pixels": total_pixels,
        "never_above_threshold": never,
        "persistent_pixels": persistent,
        "persistent_pct": (persistent / total_pixels * 100) if total_pixels else 0.0,
        "max_persistence_dates": max_persistence,
        "majority_threshold_dates": majority_threshold,
    }


def build_seasonal_summary(temporal_records: list[dict]) -> list[dict]:
    """Agrupa las fechas oficiales por lago en época seca/lluviosa y calcula
    el promedio de cyano_mean de cada grupo (8.4).

    Recibe los mismos records que produce
    src.temporal.build_temporal_summary() para no releer los raster ni
    reimplementar la lectura de cyano_mean.
    """
    rows = []
    for lake in config.LAKES:
        for season in ("seca", "lluviosa"):
            values = [
                r["cyano_mean"] for r in temporal_records
                if r["lake"] == lake and r["cyano_mean"] is not None and season_of(r["date"]) == season
            ]
            rows.append({
                "lake": lake,
                "season": season,
                "n_dates": len(values),
                "mean_cyano": float(np.mean(values)) if values else None,
            })
    return rows


def plot_seasonality(seasonal_rows: list[dict], output_path: Path | None = None) -> Path:
    """Gráfica de barras simple: promedio de cyano_mean por lago y época,
    con el tamaño de muestra (n) anotado en cada barra (8.4).
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.FIGURES_DIR / "seasonality.png")

    lakes = list(config.LAKES.keys())
    seasons = ["seca", "lluviosa"]
    x = np.arange(len(lakes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, season in enumerate(seasons):
        values, labels_n = [], []
        for lake in lakes:
            row = next((r for r in seasonal_rows if r["lake"] == lake and r["season"] == season), None)
            values.append(row["mean_cyano"] if row and row["mean_cyano"] is not None else 0)
            labels_n.append(row["n_dates"] if row else 0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=season.capitalize())
        for b, n in zip(bars, labels_n):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lake.capitalize() for lake in lakes])
    ax.set_ylabel("Índice promedio de cianobacteria")
    ax.set_title("Promedio de cianobacteria por época — comparación exploratoria")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] gráfica de estacionalidad guardada en {output_path}")
    return output_path


def build_exploratory_interpretation(
    threshold: float,
    percentage_rows: list[dict],
    persistence_summaries: dict[str, dict],
    seasonal_rows: list[dict],
) -> str:
    """Redacta la interpretación de 8.1, 8.2, 8.4 y 8.5 basada únicamente en
    los resultados calculados en este módulo.
    """
    lines = ["## Ejercicio 8: análisis exploratorio adicional\n"]
    lines.append(
        f"**Criterio de 'cianobacteria alta'**: percentil {HIGH_CYANO_PERCENTILE} calculado "
        f"sobre todos los valores válidos de cianobacteria de ambos lagos y las 22 fechas "
        f"oficiales (umbral = {threshold:.2f})."
    )

    lines.append("\n### 8.1 Extensión espacial de la floración\n")
    for lake in config.LAKES:
        lake_pct = [
            r["high_cyano_percentage"] for r in percentage_rows
            if r["lake"] == lake and r["high_cyano_percentage"] is not None
        ]
        if not lake_pct:
            continue
        max_row = max(
            (r for r in percentage_rows if r["lake"] == lake and r["high_cyano_percentage"] is not None),
            key=lambda r: r["high_cyano_percentage"],
        )
        lines.append(
            f"- **{lake.capitalize()}**: el porcentaje de área con cianobacteria alta varió entre "
            f"{min(lake_pct):.1f}% y {max(lake_pct):.1f}% según la fecha, con un promedio de "
            f"{np.mean(lake_pct):.1f}%. La mayor extensión se observó el {max_row['date']} "
            f"({max_row['high_cyano_percentage']:.1f}% del lago)."
        )

    lines.append("\n### 8.2 Persistencia espacial\n")
    for lake in config.LAKES:
        s = persistence_summaries.get(lake)
        if not s:
            continue
        lines.append(
            f"- **{lake.capitalize()}**: {s['persistent_pct']:.1f}% de los píxeles del lago "
            f"superaron el umbral en al menos {s['majority_threshold_dates']:.0f} de sus "
            f"{len(config.DATES[lake])} fechas oficiales (persistencia mayoritaria). El píxel con "
            f"mayor persistencia superó el umbral en {s['max_persistence_dates']} fecha(s). "
            f"Ver `persistencia_{lake}.png` para ubicar visualmente estas zonas."
        )
    lines.append(
        "- Nota: la nubosidad varía entre fechas, así que un punto puede aparecer con baja "
        "persistencia simplemente por tener menos fechas válidas, no solo por menor "
        "concentración real de cianobacteria."
    )

    lines.append("\n### 8.4 Patrón estacional\n")
    lines.append(
        "- Agrupación exploratoria por época del año en Guatemala (seca: noviembre-abril; "
        "lluviosa: mayo-octubre), aplicada únicamente a las fechas oficiales ya disponibles del "
        "laboratorio."
    )
    for lake in config.LAKES:
        seca = next((r for r in seasonal_rows if r["lake"] == lake and r["season"] == "seca"), None)
        lluviosa = next((r for r in seasonal_rows if r["lake"] == lake and r["season"] == "lluviosa"), None)
        if seca and lluviosa and seca["mean_cyano"] is not None and lluviosa["mean_cyano"] is not None:
            mayor = "seca" if seca["mean_cyano"] > lluviosa["mean_cyano"] else "lluviosa"
            lines.append(
                f"- **{lake.capitalize()}**: promedio en época seca = {seca['mean_cyano']:.2f} "
                f"(n={seca['n_dates']}), en época lluviosa = {lluviosa['mean_cyano']:.2f} "
                f"(n={lluviosa['n_dates']}). El promedio observado fue mayor en época {mayor}."
            )
    lines.append(
        "- Advertencia importante: solo hay 11 fechas oficiales por lago, repartidas de forma "
        "no uniforme entre ambas épocas (ver los conteos 'n=' arriba); esta comparación es "
        "únicamente descriptiva y no debe leerse como un patrón estacional estadísticamente "
        "confirmado."
    )

    lines.append("\n### 8.5 Conclusión del análisis exploratorio\n")
    lines.append(
        "- Con los datos disponibles, la floración de cianobacteria en ambos lagos no es "
        "constante en el espacio ni en el tiempo: la extensión (8.1) varía notablemente entre "
        "fechas, y solo una fracción del área de cada lago se mantiene consistentemente sobre "
        "el umbral (8.2). Con solo 11 fechas por lago no es posible confirmar un patrón "
        "estacional robusto (8.4), aunque los promedios por época ofrecen una primera señal a "
        "explorar con más muestreos."
    )

    return "\n".join(lines)


def run_exploratory_analysis(temporal_records: list[dict] | None = None) -> dict:
    """Punto de entrada único del ejercicio 8 (8.1, 8.2, 8.4, 8.5).

    Si no se recibe temporal_records (de src.temporal.build_temporal_summary()),
    los calcula aquí mismo para no depender de que el llamador ya los tenga.
    """
    config.ensure_output_dirs()

    if temporal_records is None:
        from src.temporal import build_temporal_summary
        temporal_records = build_temporal_summary()

    threshold = compute_global_threshold()

    percentage_rows = compute_high_cyano_percentage(threshold)
    save_high_cyano_percentage_csv(percentage_rows)
    affected_area_path = plot_affected_area_evolution(percentage_rows)

    persistence_summaries = {}
    persistence_paths = {}
    for lake in config.LAKES:
        persistence, valid_counts, meta = compute_pixel_persistence(lake, threshold)
        persistence_paths[lake] = plot_persistence_map(
            lake, persistence, valid_counts, meta, config.FIGURES_DIR / f"persistencia_{lake}.png"
        )
        persistence_summaries[lake] = summarize_persistence(lake, persistence, valid_counts)

    seasonal_rows = build_seasonal_summary(temporal_records)
    seasonality_path = plot_seasonality(seasonal_rows)

    interpretation = build_exploratory_interpretation(
        threshold, percentage_rows, persistence_summaries, seasonal_rows
    )

    print("\n" + interpretation)

    return {
        "threshold": threshold,
        "percentage_rows": percentage_rows,
        "affected_area_path": affected_area_path,
        "persistence_summaries": persistence_summaries,
        "persistence_paths": persistence_paths,
        "seasonal_rows": seasonal_rows,
        "seasonality_path": seasonality_path,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    run_exploratory_analysis()