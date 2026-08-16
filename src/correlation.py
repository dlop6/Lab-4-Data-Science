"""Correlacion entre cianobacteria y los indices NDVI/NDWI.

Ejercicio 6 del laboratorio.

Para cada lago:
- usa solamente pixeles validos coincidentes de cyano, NDVI y NDWI;
- limita los pixeles al contorno real del lago usando el GeoJSON;
- acumula los datos de las 11 fechas oficiales;
- calcula correlacion de Pearson Cyano-NDVI y Cyano-NDWI;
- genera cuatro scatterplots;
- guarda los resultados en outputs/tables/correlations.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.spatial import load_masked_raster


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

CORRELATION_FIGURES_DIR = config.FIGURES_DIR / "correlation"
CORRELATIONS_CSV = config.TABLES_DIR / "correlations.csv"

# Los coeficientes se calculan con todos los pixeles validos.
# Para los scatterplots se usa una muestra para evitar graficar millones
# de puntos y producir figuras innecesariamente pesadas.
MAX_SCATTER_POINTS = 100_000
RANDOM_SEED = 42


def load_valid_pixels(lake: str, date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga los tres indices de una fecha y conserva solo pixeles validos.

    La mascara valida es exactamente:

        finite(cyano)
        AND finite(ndvi)
        AND finite(ndwi)
        AND dentro del lago

    El ultimo requisito se cumple mediante load_masked_raster(), que convierte
    en NaN todo lo que se encuentre fuera del GeoJSON del lago.

    Returns:
        Tupla (cyano, ndvi, ndwi) con vectores 1D de pixeles coincidentes.
    """
    raster_dir = config.raster_dir(lake, date)

    cyano_path = raster_dir / "cyano.tif"
    ndvi_path = raster_dir / "ndvi.tif"
    ndwi_path = raster_dir / "ndwi.tif"

    for path in (cyano_path, ndvi_path, ndwi_path):
        if not path.exists():
            raise FileNotFoundError(f"no existe el raster requerido: {path}")

    cyano, _ = load_masked_raster(lake, cyano_path)
    ndvi, _ = load_masked_raster(lake, ndvi_path)
    ndwi, _ = load_masked_raster(lake, ndwi_path)

    if cyano.shape != ndvi.shape or cyano.shape != ndwi.shape:
        raise ValueError(
            f"los raster de {lake} {date} no tienen las mismas dimensiones: "
            f"cyano={cyano.shape}, ndvi={ndvi.shape}, ndwi={ndwi.shape}"
        )

    valid_mask = (
        np.isfinite(cyano)
        & np.isfinite(ndvi)
        & np.isfinite(ndwi)
    )

    return (
        cyano[valid_mask],
        ndvi[valid_mask],
        ndwi[valid_mask],
    )


def collect_lake_pixels(lake: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acumula los pixeles validos coincidentes de las 11 fechas de un lago."""
    cyano_values = []
    ndvi_values = []
    ndwi_values = []

    for date in config.DATES[lake]:
        cyano, ndvi, ndwi = load_valid_pixels(lake, date)

        if cyano.size == 0:
            print(f"[warning] {lake} {date}: no hay pixeles validos coincidentes")
            continue

        cyano_values.append(cyano)
        ndvi_values.append(ndvi)
        ndwi_values.append(ndwi)

        print(
            f"[ok] {lake} {date}: "
            f"{cyano.size:,} pixeles validos coincidentes"
        )

    if not cyano_values:
        raise ValueError(f"no hay pixeles validos para {lake}")

    return (
        np.concatenate(cyano_values),
        np.concatenate(ndvi_values),
        np.concatenate(ndwi_values),
    )


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calcula la correlacion de Pearson entre dos vectores."""
    if x.size != y.size:
        raise ValueError("los vectores deben tener la misma cantidad de elementos")

    if x.size < 2:
        raise ValueError("se necesitan al menos dos observaciones para correlacion")

    if np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("no se puede calcular correlacion con una variable constante")

    return float(np.corrcoef(x, y)[0, 1])


def sample_for_scatter(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = MAX_SCATTER_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene una muestra reproducible para visualizar el scatterplot.

    La correlacion NO se calcula con esta muestra. Esta funcion se utiliza
    unicamente para que la figura sea manejable cuando existen millones
    de pixeles validos.
    """
    if x.size <= max_points:
        return x, y

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(x.size, size=max_points, replace=False)

    return x[indices], y[indices]


def plot_scatter(
    lake: str,
    index_name: str,
    index_values: np.ndarray,
    cyano_values: np.ndarray,
    correlation: float,
    output_path: Path,
) -> Path:
    """Genera un scatterplot Cyano vs NDVI o Cyano vs NDWI."""
    x_plot, y_plot = sample_for_scatter(index_values, cyano_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        x_plot,
        y_plot,
        s=5,
        alpha=0.25,
    )

    ax.set_xlabel(index_name.upper())
    ax.set_ylabel("Índice de cianobacteria")
    ax.set_title(
        f"{lake.capitalize()}: Cianobacteria vs {index_name.upper()}\n"
        f"Pearson r = {correlation:.4f}"
    )

    ax.grid(alpha=0.2)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] scatterplot guardado en {output_path}")

    return output_path


def save_correlations_csv(rows: list[dict], output_path: Path) -> Path:
    """Guarda la tabla final de correlaciones."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lake", "index", "correlation"],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] tabla de correlaciones guardada en {output_path}")

    return output_path


def run_correlation_analysis() -> list[dict]:
    """Ejecuta todo el analisis de correlacion del ejercicio 6.

    Genera:
    - cuatro coeficientes de Pearson;
    - cuatro scatterplots;
    - outputs/tables/correlations.csv.

    Returns:
        Lista con los cuatro resultados de correlacion.
    """
    config.ensure_output_dirs()
    CORRELATION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for lake in config.LAKES:
        print("\n" + "=" * 60)
        print(f"CORRELACIONES: {lake.upper()}")
        print("=" * 60)

        cyano, ndvi, ndwi = collect_lake_pixels(lake)

        print(
            f"[info] {lake}: "
            f"{cyano.size:,} pixeles acumulados de las 11 fechas"
        )

        correlation_ndvi = pearson_correlation(cyano, ndvi)
        correlation_ndwi = pearson_correlation(cyano, ndwi)

        print(f"[resultado] Cyano-NDVI: r={correlation_ndvi:.6f}")
        print(f"[resultado] Cyano-NDWI: r={correlation_ndwi:.6f}")

        results.append(
            {
                "lake": lake,
                "index": "ndvi",
                "correlation": correlation_ndvi,
            }
        )

        results.append(
            {
                "lake": lake,
                "index": "ndwi",
                "correlation": correlation_ndwi,
            }
        )

        plot_scatter(
            lake=lake,
            index_name="ndvi",
            index_values=ndvi,
            cyano_values=cyano,
            correlation=correlation_ndvi,
            output_path=(
                CORRELATION_FIGURES_DIR
                / f"{lake}_cyano_ndvi.png"
            ),
        )

        plot_scatter(
            lake=lake,
            index_name="ndwi",
            index_values=ndwi,
            cyano_values=cyano,
            correlation=correlation_ndwi,
            output_path=(
                CORRELATION_FIGURES_DIR
                / f"{lake}_cyano_ndwi.png"
            ),
        )

    save_correlations_csv(results, CORRELATIONS_CSV)

    return results


if __name__ == "__main__":
    run_correlation_analysis()