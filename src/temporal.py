"""Análisis temporal y visualización (ejercicio 4).

Este módulo:
- lee lagos y fechas desde config.py (única fuente de verdad, no duplica
  listas ni rutas);
- lee data/rasters/{lake}/{date}/cyano.tif generados por indices.py,
  ignorando NoData/NaN (no reimplementa el cálculo del índice, ni toca
  Sentinel);
- calcula cyano_mean y valid_pixels por lago y fecha;
- genera outputs/tables/temporal_summary.csv con las columnas exactas del
  contrato (lake,date,cyano_mean,valid_pixels);
- genera outputs/figures/temporal_{lake}.png;
- identifica picos de floración y fechas críticas;
- redacta una interpretación breve basada únicamente en los resultados
  observados.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import config

CYANO_FILENAME = "cyano.tif"

# Umbral solo para *advertir* en la interpretación que una fecha tiene poca
# cobertura válida (no filtra ni descarta datos). El cyano.tif ya trae una
# mascara de agua (waterBody() en indices.py), asi que valid_pixels/total
# ronda ~20-25% en TODAS las fechas de forma normal (el resto es tierra
# alrededor del lago dentro del bbox rectangular) -- eso no es una alerta.
# Lo que sí importa es cuánta agua del lago quedó cubierta en cada fecha
# respecto a las demás fechas del mismo lago (nubes, bordes de escena,
# etc.), así que el umbral se aplica sobre esa cobertura relativa. 0.75
# deja pasar el caso ya documentado en el pdf (Amatitlán 2026-02-07, ~57.1%
# de cobertura válida respecto al resto de fechas de ese lago).
LOW_COVERAGE_WARNING_RATIO = 0.75


def read_cyano_mean(lake: str, date: str) -> dict:
    """Lee cyano.tif de una fecha/lago y calcula su media ignorando NaN/NoData.

    No lanza excepción si el raster no existe todavía (por ejemplo si Persona
    2 aún no ha generado esa fecha); en ese caso regresa cyano_mean=None para
    que el resto del pipeline lo pueda saltar sin romperse.
    """
    path = config.raster_dir(lake, date) / CYANO_FILENAME
    if not path.exists():
        print(f"[warn] no existe {path}, se omite esta fecha")
        return {
            "lake": lake,
            "date": date,
            "cyano_mean": None,
            "valid_pixels": 0,
            "total_pixels": 0,
        }

    with rasterio.open(path) as src:
        data = src.read(1).astype("float64", copy=False)
        nodata = src.nodata

    valid_mask = np.isfinite(data)
    if nodata is not None:
        valid_mask &= data != nodata

    valid_pixels = int(valid_mask.sum())
    total_pixels = int(data.size)
    cyano_mean = float(data[valid_mask].mean()) if valid_pixels > 0 else None

    return {
        "lake": lake,
        "date": date,
        "cyano_mean": cyano_mean,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }


def build_temporal_summary() -> list[dict]:
    """Recorre ambos lagos y sus fechas oficiales (config.DATES) y arma la
    lista de registros, uno por lago/fecha, ordenada cronológicamente.
    """
    records = [
        read_cyano_mean(lake, date)
        for lake in config.LAKES
        for date in config.DATES[lake]
    ]
    records.sort(key=lambda r: (r["lake"], r["date"]))
    return records


def save_temporal_summary_csv(records: list[dict], output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/temporal_summary.csv con exactamente las columnas
    lake,date,cyano_mean,valid_pixels (contrato del repo).
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.TABLES_DIR / "temporal_summary.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(config.TEMPORAL_SUMMARY_COLUMNS)
        for r in records:
            mean_str = "" if r["cyano_mean"] is None else f"{r['cyano_mean']:.6f}"
            writer.writerow([r["lake"], r["date"], mean_str, r["valid_pixels"]])

    print(f"[ok] tabla temporal guardada en {output_path}")
    return output_path


def plot_temporal_series(records: list[dict], lake: str, output_path: Path | None = None) -> Path:
    """Genera y guarda el gráfico de línea de cyano_mean vs fecha para un lago,
    marcando el pico máximo observado.
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.FIGURES_DIR / f"temporal_{lake}.png")

    lake_records = [r for r in records if r["lake"] == lake and r["cyano_mean"] is not None]
    dates = [r["date"] for r in lake_records]
    values = [r["cyano_mean"] for r in lake_records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, values, marker="o", color="#2a7f62", linewidth=2)

    if values:
        peak_idx = int(np.argmax(values))
        ax.scatter(
            [dates[peak_idx]], [values[peak_idx]],
            color="crimson", zorder=5, s=80,
            label=f"pico: {dates[peak_idx]} ({values[peak_idx]:.2f})",
        )
        ax.legend()

    ax.set_title(f"Índice promedio de cianobacteria por fecha — {lake.capitalize()}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Índice de cianobacteria (promedio)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] gráfica guardada en {output_path}")
    return output_path


def identify_peaks(records: list[dict], lake: str) -> dict:
    """Calcula máximo, mínimo y sus fechas para un lago, a partir de las
    fechas con cyano_mean válido.
    """
    lake_records = [r for r in records if r["lake"] == lake and r["cyano_mean"] is not None]
    if not lake_records:
        return {"lake": lake, "max_value": None, "min_value": None, "n_dates": 0}

    max_r = max(lake_records, key=lambda r: r["cyano_mean"])
    min_r = min(lake_records, key=lambda r: r["cyano_mean"])

    return {
        "lake": lake,
        "max_value": max_r["cyano_mean"],
        "max_date": max_r["date"],
        "min_value": min_r["cyano_mean"],
        "min_date": min_r["date"],
        "mean_of_means": float(np.mean([r["cyano_mean"] for r in lake_records])),
        "n_dates": len(lake_records),
    }


def build_interpretation(records: list[dict]) -> str:
    """Redacta una interpretación breve basada únicamente en los valores
    calculados (rango, promedio, fecha del pico) más una nota de cobertura
    para fechas con muchos píxeles inválidos.
    """
    lines = []
    for lake in config.LAKES:
        stats = identify_peaks(records, lake)
        if stats["max_value"] is None:
            lines.append(f"- {lake.capitalize()}: no hay datos válidos para interpretar todavía.")
            continue
        lines.append(
            f"- {lake.capitalize()}: el índice promedio de cianobacteria varió entre "
            f"{stats['min_value']:.2f} ({stats['min_date']}) y {stats['max_value']:.2f} "
            f"({stats['max_date']}), con un promedio de {stats['mean_of_means']:.2f} a lo "
            f"largo de {stats['n_dates']} fechas."
        )

    # Cobertura relativa: se compara valid_pixels de cada fecha contra el
    # máximo de valid_pixels visto en ese mismo lago (referencia de "agua
    # del lago totalmente cubierta"), no contra el total del raster.
    for lake in config.LAKES:
        lake_records = [r for r in records if r["lake"] == lake and r["valid_pixels"] > 0]
        if not lake_records:
            continue
        reference = max(r["valid_pixels"] for r in lake_records)
        if reference == 0:
            continue
        for r in lake_records:
            ratio = r["valid_pixels"] / reference
            if ratio < LOW_COVERAGE_WARNING_RATIO:
                lines.append(
                    f"- Nota: {r['lake'].capitalize()} {r['date']} tiene cobertura válida "
                    f"parcial (~{ratio * 100:.1f}% respecto al resto de fechas de ese lago); "
                    f"su promedio debe interpretarse con cautela."
                )

    return "\n".join(lines)


def run_temporal_analysis() -> dict:
    """Punto de entrada único de este módulo: regenera automáticamente la
    tabla temporal y ambas gráficas a partir de los raster existentes en
    data/rasters/, e imprime la interpretación inicial del ejercicio 4.
    """
    records = build_temporal_summary()
    csv_path = save_temporal_summary_csv(records)

    figure_paths = {lake: plot_temporal_series(records, lake) for lake in config.LAKES}
    peaks = {lake: identify_peaks(records, lake) for lake in config.LAKES}
    interpretation = build_interpretation(records)

    print("\n=== Interpretación inicial (ejercicio 4) ===")
    print(interpretation)

    return {
        "records": records,
        "csv_path": csv_path,
        "figure_paths": figure_paths,
        "peaks": peaks,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    run_temporal_analysis()