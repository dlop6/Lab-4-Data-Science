"""Distribuciones del indice de cianobacteria.

Ejercicio 8.3 del laboratorio.

Para cada lago:
- carga cyano.tif de las 11 fechas oficiales;
- aplica la mascara del GeoJSON del lago;
- conserva solamente valores finitos;
- genera un histograma acumulado;
- genera un boxplot comparativo entre fechas.
"""

from __future__ import annotations

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

DISTRIBUTIONS_DIR = config.FIGURES_DIR / "distributions"

# Limita solamente la cantidad de puntos utilizados para construir las
# visualizaciones. No modifica los raster originales.
MAX_VALUES_PER_DATE = 100_000
RANDOM_SEED = 42


def load_cyano_values(lake: str, date: str) -> np.ndarray:
    """Carga los valores validos de cianobacteria de un lago y fecha.

    load_masked_raster() convierte a NaN todo lo que se encuentra fuera
    del GeoJSON del lago. Luego se eliminan todos los valores no finitos.

    Returns:
        Vector 1D con los valores validos de cianobacteria.
    """
    path = config.raster_dir(lake, date) / "cyano.tif"

    if not path.exists():
        raise FileNotFoundError(f"no existe el raster requerido: {path}")

    cyano, _ = load_masked_raster(lake, path)

    values = cyano[np.isfinite(cyano)]

    if values.size == 0:
        raise ValueError(
            f"{lake} {date} no contiene valores validos de cianobacteria"
        )

    return values


def sample_values(
    values: np.ndarray,
    max_values: int = MAX_VALUES_PER_DATE,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Obtiene una muestra reproducible para las visualizaciones."""
    if values.size <= max_values:
        return values

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        values.size,
        size=max_values,
        replace=False,
    )

    return values[indices]


def collect_lake_distributions(
    lake: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Carga las distribuciones de las 11 fechas oficiales de un lago."""
    dates = []
    distributions = []

    for i, date in enumerate(config.DATES[lake]):
        values = load_cyano_values(lake, date)

        sampled = sample_values(
            values,
            seed=RANDOM_SEED + i,
        )

        dates.append(date)
        distributions.append(sampled)

        print(
            f"[ok] {lake} {date}: "
            f"{values.size:,} valores validos | "
            f"{sampled.size:,} usados en visualizacion"
        )

    return dates, distributions


def plot_histogram(
    lake: str,
    distributions: list[np.ndarray],
    output_path: Path,
) -> Path:
    """Genera un histograma de cianobacteria para un lago.

    Combina los valores muestreados de las 11 fechas para representar
    la distribucion general observada durante el periodo estudiado.
    """
    all_values = np.concatenate(distributions)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(
        all_values,
        bins=50,
        edgecolor="black",
        alpha=0.75,
    )

    ax.set_title(
        f"{lake.capitalize()}: distribución del índice de cianobacteria"
    )
    ax.set_xlabel("Índice de cianobacteria")
    ax.set_ylabel("Frecuencia")

    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] histograma guardado en {output_path}")

    return output_path


def plot_boxplot(
    lake: str,
    dates: list[str],
    distributions: list[np.ndarray],
    output_path: Path,
) -> Path:
    """Genera boxplots de cianobacteria comparando las 11 fechas."""
    fig, ax = plt.subplots(figsize=(13, 7))

    ax.boxplot(
        distributions,
        tick_labels=dates,
        showfliers=False,
    )

    ax.set_title(
        f"{lake.capitalize()}: distribución de cianobacteria por fecha"
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Índice de cianobacteria")

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] boxplot guardado en {output_path}")

    return output_path


def run_distribution_analysis() -> list[Path]:
    """Ejecuta el analisis de distribuciones del ejercicio 8.3.

    Genera para cada lago:
    - un histograma general de cianobacteria;
    - un boxplot comparativo entre las 11 fechas.

    Returns:
        Lista con las cuatro figuras generadas.
    """
    config.ensure_output_dirs()
    DISTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []

    for lake in config.LAKES:
        print("\n" + "=" * 60)
        print(f"DISTRIBUCIONES: {lake.upper()}")
        print("=" * 60)

        dates, distributions = collect_lake_distributions(lake)

        histogram_path = (
            DISTRIBUTIONS_DIR
            / f"histograma_{lake}.png"
        )

        boxplot_path = (
            DISTRIBUTIONS_DIR
            / f"boxplot_{lake}.png"
        )

        plot_histogram(
            lake=lake,
            distributions=distributions,
            output_path=histogram_path,
        )

        plot_boxplot(
            lake=lake,
            dates=dates,
            distributions=distributions,
            output_path=boxplot_path,
        )

        outputs.extend(
            [
                histogram_path,
                boxplot_path,
            ]
        )

    return outputs


if __name__ == "__main__":
    run_distribution_analysis()