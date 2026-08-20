"""Conversion de datos Sentinel-2 a formato tabular.

Genera un raster multibanda con:
- B02
- B03
- B04
- B08
- dataMask

Luego convierte cada pixel valido dentro del lago en una fila:

lake | date | b2 | b3 | b4 | b8 | longitude | latitude | ndvi | ndwi
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

import config
from src.sentinel import request_product


# ---------------------------------------------------------------------------
# EvalScript
# ---------------------------------------------------------------------------

BANDS_EVALSCRIPT = r"""
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B08", "dataMask"]
        }],
        output: {
            bands: 5,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(s) {
    return [
        s.B02,
        s.B03,
        s.B04,
        s.B08,
        s.dataMask
    ];
}
"""


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

TABULAR_DIR = config.DATA_DIR / "tabular"


def bands_raster_path(lake: str, date: str) -> Path:
    """Ruta del raster multibanda de una fecha."""
    return config.raster_dir(lake, date) / "bands.tif"


def tabular_csv_path(lake: str, date: str) -> Path:
    """Ruta del CSV tabular de una fecha."""
    return TABULAR_DIR / lake / f"{date}.csv"


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def download_bands(
    lake: str,
    date: str,
    skip_existing: bool = True,
) -> Path:
    """Descarga B02, B03, B04 y B08 en un unico GeoTIFF.

    La quinta banda corresponde a dataMask y se utiliza solamente para
    determinar que pixeles tienen datos validos.
    """
    output_path = bands_raster_path(lake, date)

    if skip_existing and output_path.exists():
        print(f"[skip] ya existe {output_path}")
        return output_path

    print(f"[request] {lake} {date}: B02 B03 B04 B08")

    request_product(
        lake,
        date,
        BANDS_EVALSCRIPT,
        output_path,
    )

    print(f"[ok] raster multibanda guardado en {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def load_lake_geometry(lake: str) -> list[dict]:
    """Carga las geometrías del GeoJSON correspondiente al lago."""
    import json

    geojson_path = config.GEOJSON_PATHS[lake]

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        return [
            feature["geometry"]
            for feature in geojson["features"]
        ]

    if geojson.get("type") == "Feature":
        return [geojson["geometry"]]

    return [geojson]


# ---------------------------------------------------------------------------
# Raster -> DataFrame
# ---------------------------------------------------------------------------

def raster_to_dataframe(
    lake: str,
    date: str,
    raster_path: Path | None = None,
) -> pd.DataFrame:
    """Convierte un raster multibanda en un DataFrame.

    Solo conserva pixeles que:
    - tienen dataMask valido;
    - contienen valores finitos en B02/B03/B04/B08;
    - se encuentran dentro del GeoJSON del lago.

    Returns:
        DataFrame con columnas:

        lake, date, b2, b3, b4, b8,
        longitude, latitude, ndvi, ndwi
    """
    if raster_path is None:
        raster_path = bands_raster_path(lake, date)

    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"no existe el raster multibanda: {raster_path}"
        )

    geometries = load_lake_geometry(lake)

    with rasterio.open(raster_path) as src:
        if src.count != 5:
            raise ValueError(
                f"{raster_path} debe tener 5 bandas; tiene {src.count}"
            )

        b2 = src.read(1).astype("float32")
        b3 = src.read(2).astype("float32")
        b4 = src.read(3).astype("float32")
        b8 = src.read(4).astype("float32")
        data_mask = src.read(5)

        # True dentro del poligono del lago.
        lake_mask = geometry_mask(
            geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )

        valid = (
            lake_mask
            & (data_mask > 0)
            & np.isfinite(b2)
            & np.isfinite(b3)
            & np.isfinite(b4)
            & np.isfinite(b8)
        )

        rows, cols = np.where(valid)

        # Coordenadas del centro de cada pixel.
        xs, ys = rasterio.transform.xy(
            src.transform,
            rows,
            cols,
            offset="center",
        )

    # Extraemos solo los valores validos.
    b2_values = b2[valid]
    b3_values = b3[valid]
    b4_values = b4[valid]
    b8_values = b8[valid]

    # ---------------------------------------------------------------
    # NDVI
    # ---------------------------------------------------------------

    ndvi_den = b8_values + b4_values

    ndvi = np.full(
        b8_values.shape,
        np.nan,
        dtype="float32",
    )

    np.divide(
        b8_values - b4_values,
        ndvi_den,
        out=ndvi,
        where=ndvi_den != 0,
    )

    # ---------------------------------------------------------------
    # NDWI
    # ---------------------------------------------------------------

    ndwi_den = b3_values + b8_values

    ndwi = np.full(
        b3_values.shape,
        np.nan,
        dtype="float32",
    )

    np.divide(
        b3_values - b8_values,
        ndwi_den,
        out=ndwi,
        where=ndwi_den != 0,
    )

    # ---------------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------------

    df = pd.DataFrame(
        {
            "lake": lake,
            "date": date,
            "b2": b2_values,
            "b3": b3_values,
            "b4": b4_values,
            "b8": b8_values,
            "longitude": np.asarray(xs),
            "latitude": np.asarray(ys),
            "ndvi": ndvi,
            "ndwi": ndwi,
        }
    )

    return df


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------

def save_dataframe_csv(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Guarda un DataFrame en CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"[ok] {len(df):,} filas guardadas en {output_path}"
    )

    return output_path


# ---------------------------------------------------------------------------
# Procesamiento de una fecha
# ---------------------------------------------------------------------------

def process_date(
    lake: str,
    date: str,
    skip_existing_raster: bool = True,
) -> pd.DataFrame:
    """Descarga las bandas y convierte una fecha a formato tabular."""
    raster_path = download_bands(
        lake,
        date,
        skip_existing=skip_existing_raster,
    )

    df = raster_to_dataframe(
        lake,
        date,
        raster_path,
    )

    output_path = tabular_csv_path(lake, date)

    save_dataframe_csv(
        df,
        output_path,
    )

    return df


if __name__ == "__main__":
    # Prueba inicial pequeña: una fecha de Atitlan.
    df = process_date(
        "atitlan",
        "2025-01-18",
    )

    print()
    print(df.head())
    print()
    print(df.info())