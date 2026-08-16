"""Analisis espacial (ejercicio 5): mapas de cianobacteria por lago y fecha,
enmascarados con el contorno real del lago (geojson) para excluir tierra.

Este modulo:
- lee lagos y fechas desde config.py (misma fuente de verdad que los demas);
- reutiliza src.temporal.read_cyano_mean() solo para decidir que fechas
  representan un valor bajo/intermedio/alto (no reimplementa ese calculo);
- no toca sentinel, ni las formulas de ndvi/ndwi/cyano de indices.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask

import config
from src.temporal import read_cyano_mean

CYANO_FILENAME = "cyano.tif"

# recorte de outliers para la escala de color: con percentil 2-98 un pico
# puntual (ej. ~96 en una sola fecha de atitlan) no aplasta el resto de
# mapas de ese lago a un solo color. valor ya decidido, no configurable.
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98


def load_masked_raster(lake: str, tif_path: Path) -> tuple[np.ndarray, dict]:
    """abre un raster y deja NaN todo lo que caiga fuera del contorno real
    del lago (geojson de config.GEOJSON_PATHS). funciona igual para cyano,
    ndvi o ndwi -- es la unica funcion de mascara, no hay una version por
    indice.
    """
    if lake not in config.LAKES:
        raise ValueError(f"lago desconocido: {lake}. debe ser uno de {list(config.LAKES)}")

    geojson_path = config.GEOJSON_PATHS[lake]
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)
    geometry = geojson["features"][0]["geometry"]

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"{tif_path} no tiene CRS, no se puede enmascarar")
        # el geojson se genero en WGS84 (EPSG:4326) y los raster tambien se
        # guardan en EPSG:4326 (ver src/sentinel.py::_save_tiff), asi que no
        # hace falta reproyectar. si algun dia dejan de coincidir, mejor
        # fallar aqui con un mensaje claro que enmascarar mal en silencio.
        if str(src.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
            raise ValueError(
                f"{tif_path} esta en {src.crs}, se esperaba EPSG:4326 igual que "
                f"el geojson de {lake}; falta reproyectar antes de enmascarar"
            )

        masked, transform = rasterio.mask.mask(
            src, [geometry], nodata=np.nan, filled=True
        )
        meta = {
            "transform": transform,
            "crs": src.crs,
            "bounds": src.bounds,
        }

    # mask() regresa (bands, h, w); cyano/ndvi/ndwi son de 1 banda
    return masked[0], meta


def compute_color_scale(lake: str) -> tuple[float, float]:
    """calcula (vmin, vmax) para el lago completo, una sola vez, a partir de
    los pixeles validos de cyano.tif de sus 11 fechas oficiales. se reusa
    en todos los mapas individuales y en el comparativo de ese lago para
    que la escala de color sea consistente.
    """
    all_valid_pixels = []
    for date in config.DATES[lake]:
        tif_path = config.raster_dir(lake, date) / CYANO_FILENAME
        if not tif_path.exists():
            continue
        array, _ = load_masked_raster(lake, tif_path)
        valid = array[np.isfinite(array)]
        if valid.size > 0:
            all_valid_pixels.append(valid)

    if not all_valid_pixels:
        raise ValueError(f"no hay ningun cyano.tif valido para {lake} todavia")

    combined = np.concatenate(all_valid_pixels)
    vmin = float(np.percentile(combined, PERCENTILE_LOW))
    vmax = float(np.percentile(combined, PERCENTILE_HIGH))
    return vmin, vmax


def plot_cyano_map(lake: str, date: str, vmin: float, vmax: float, output_path: Path) -> Path:
    """genera y guarda el mapa de cianobacteria de una fecha, con tierra
    excluida por la mascara del geojson y la escala de color fija del lago.
    """
    tif_path = config.raster_dir(lake, date) / CYANO_FILENAME
    array, meta = load_masked_raster(lake, tif_path)
    bounds = meta["bounds"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        array,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    )
    ax.set_title(f"Índice de cianobacteria — {lake.capitalize()} — {date}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.colorbar(im, ax=ax, label="Índice de cianobacteria")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] mapa espacial guardado en {output_path}")
    return output_path


def select_comparison_dates(lake: str) -> dict:
    """elige, de las fechas oficiales del lago con cyano_mean valido, la de
    valor bajo (minimo), alto (maximo) e intermedio (la mas cercana a la
    mediana). criterio fijo, no depende de eleccion manual.
    """
    records = []
    for date in config.DATES[lake]:
        info = read_cyano_mean(lake, date)
        if info["cyano_mean"] is not None:
            records.append((date, info["cyano_mean"]))

    if len(records) < 3:
        raise ValueError(
            f"{lake} solo tiene {len(records)} fecha(s) con cyano_mean valido; "
            f"hacen falta al menos 3 para elegir bajo/intermedio/alto"
        )

    records.sort(key=lambda r: r[1])
    low = records[0]
    high = records[-1]

    values = [r[1] for r in records]
    median = float(np.median(values))
    mid = min(records, key=lambda r: abs(r[1] - median))

    return {"bajo": low, "intermedio": mid, "alto": high}


def generate_comparison_map(lake: str, vmin: float, vmax: float, output_path: Path) -> dict:
    """genera la figura comparativa de 3 fechas (bajo/intermedio/alto) de un
    lago, con la misma escala de color y el mismo estilo que los mapas
    individuales.
    """
    chosen = select_comparison_dates(lake)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    im = None
    for ax, (label, (date, value)) in zip(axes, chosen.items()):
        tif_path = config.raster_dir(lake, date) / CYANO_FILENAME
        array, meta = load_masked_raster(lake, tif_path)
        bounds = meta["bounds"]
        im = ax.imshow(
            array,
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        )
        ax.set_title(f"{label} — {date}\n(promedio {value:.2f})")
        ax.set_xlabel("Longitud")

    axes[0].set_ylabel("Latitud")
    fig.suptitle(f"Comparativo de cianobacteria — {lake.capitalize()}")
    fig.colorbar(im, ax=axes, label="Índice de cianobacteria", shrink=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] comparativo guardado en {output_path}")
    return chosen


def generate_all_spatial_maps() -> dict:
    """punto de entrada unico del ejercicio 5: genera los mapas individuales
    de las 11 fechas de cada lago y el comparativo de 3 fechas por lago.

    insumos para la seccion "analisis espacial" del informe (ejercicio 5),
    a redactar aparte, dirigido a lectores sin conocimientos de programacion:
    - los mapas individuales (uno por fecha) muestran donde se concentran
      los valores altos en cada fecha;
    - el comparativo bajo/intermedio/alto de cada lago muestra el cambio
      espacial entre un momento de poca y mucha cianobacteria;
    - si a simple vista una misma zona aparece con valores altos en mas de
      una de las 3 fechas del comparativo, eso es solo una observacion
      cualitativa del ejercicio 5 -- el calculo formal de persistencia
      (ejercicio 8.2) le corresponde a persona 3, no se implementa aca;
    - limitacion a mencionar si amatitlan/2026-02-07 queda elegida en el
      comparativo: esa fecha tiene cobertura valida parcial (~57.1%, ver
      config.py y src/temporal.py::LOW_COVERAGE_WARNING_RATIO), su
      promedio es menos confiable que el de una fecha con cobertura completa.
    """
    config.ensure_output_dirs()

    individual_maps: list[Path] = []
    comparison_maps: dict[str, dict] = {}

    for lake in config.LAKES:
        vmin, vmax = compute_color_scale(lake)
        print(f"[info] escala de color {lake}: vmin={vmin:.4f} vmax={vmax:.4f}")

        out_dir = config.spatial_figures_dir(lake)
        for date in config.DATES[lake]:
            tif_path = config.raster_dir(lake, date) / CYANO_FILENAME
            if not tif_path.exists():
                print(f"[skip] {lake} {date}: no existe {CYANO_FILENAME} todavia")
                continue
            output_path = out_dir / f"cyano_{date}.png"
            plot_cyano_map(lake, date, vmin, vmax, output_path)
            individual_maps.append(output_path)

        comparison_path = config.SPATIAL_FIGURES_DIR / f"comparativo_{lake}.png"
        chosen = generate_comparison_map(lake, vmin, vmax, comparison_path)
        comparison_maps[lake] = {
            "path": comparison_path,
            "dates": {label: {"date": d, "cyano_mean": v} for label, (d, v) in chosen.items()},
        }

    return {"individual_maps": individual_maps, "comparison_maps": comparison_maps}


if __name__ == "__main__":
    generate_all_spatial_maps()
