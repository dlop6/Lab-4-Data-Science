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
from rasterio.warp import reproject, Resampling

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
            bands: ["B02", "B03", "B04", "B08", "SCL", "dataMask"]
        }],
        output: {
            bands: 6,
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
        s.SCL,
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

    Las bandas adicionales son:
    - SCL: Scene Classification Layer, usada para filtrar nubes y pixeles
    problematicos.
    - dataMask: indica disponibilidad valida de datos.

    SCL y dataMask se usan solamente para limpieza y no se incluyen como
    variables finales del dataset.
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
    """Convierte un raster multibanda en un DataFrame limpio.

    Cada fila representa un pixel geografico valido dentro del lago.

    Se conservan solamente pixeles que:
    - estan dentro del GeoJSON del lago;
    - tienen dataMask valido;
    - no pertenecen a clases SCL problematicas;
    - tienen B02, B03, B04 y B08 finitos;
    - tienen un valor valido de cianobacteria.

    Returns:
        DataFrame con columnas:

        lake, date, longitude, latitude,
        b2, b3, b4, b8, ndvi, ndwi, cyano
    """
    if raster_path is None:
        raster_path = bands_raster_path(lake, date)

    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"no existe el raster multibanda: {raster_path}"
        )

    cyano_path = config.raster_dir(lake, date) / "cyano.tif"

    if not cyano_path.exists():
        raise FileNotFoundError(
            f"no existe el raster de cianobacteria: {cyano_path}"
        )

    geometries = load_lake_geometry(lake)

    with rasterio.open(raster_path) as src:
        if src.count != 6:
            raise ValueError(
                f"{raster_path} debe tener 6 bandas; tiene {src.count}"
            )

        # Bandas espectrales.
        b2 = src.read(1).astype("float32")
        b3 = src.read(2).astype("float32")
        b4 = src.read(3).astype("float32")
        b8 = src.read(4).astype("float32")

        # Capas utilizadas para limpieza.
        scl = src.read(5)
        data_mask = src.read(6)

        # -----------------------------------------------------------
        # Alinear cyano.tif con la grilla del raster de bandas
        # -----------------------------------------------------------

        cyano = np.full(
            (src.height, src.width),
            np.nan,
            dtype="float32",
        )

        with rasterio.open(cyano_path) as cyano_src:
            reproject(
                source=cyano_src.read(1),
                destination=cyano,
                src_transform=cyano_src.transform,
                src_crs=cyano_src.crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

        # -----------------------------------------------------------
        # Mascara geografica del lago
        # -----------------------------------------------------------

        lake_mask = geometry_mask(
            geometries,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )

        # -----------------------------------------------------------
        # Scene Classification Layer
        # -----------------------------------------------------------
        #
        # Clases que no queremos utilizar:
        #
        # 0  = No Data
        # 1  = Saturated / defective
        # 3  = Cloud shadows
        # 8  = Cloud medium probability
        # 9  = Cloud high probability
        # 10 = Thin cirrus
        # 11 = Snow / ice

        invalid_scl = np.isin(
            scl,
            [0, 1, 3, 8, 9, 10, 11],
        )

        # -----------------------------------------------------------
        # Mascara valida final
        # -----------------------------------------------------------

        valid = (
            lake_mask
            & (data_mask > 0)
            & (~invalid_scl)
            & np.isfinite(b2)
            & np.isfinite(b3)
            & np.isfinite(b4)
            & np.isfinite(b8)
            & np.isfinite(cyano)
        )

        rows, cols = np.where(valid)

        if rows.size == 0:
            raise ValueError(
                f"{lake} {date} no contiene pixeles validos"
            )

        # Coordenadas del centro de cada pixel.
        xs, ys = rasterio.transform.xy(
            src.transform,
            rows,
            cols,
            offset="center",
        )

        # Extraer solamente pixeles validos.
        b2_values = b2[valid]
        b3_values = b3[valid]
        b4_values = b4[valid]
        b8_values = b8[valid]
        cyano_values = cyano[valid]

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
            "longitude": np.asarray(xs),
            "latitude": np.asarray(ys),
            "b2": b2_values,
            "b3": b3_values,
            "b4": b4_values,
            "b8": b8_values,
            "ndvi": ndvi,
            "ndwi": ndwi,
            "cyano": cyano_values,
        }
    )

    # Eliminar observaciones donde los indices no pudieron calcularse
    # debido a denominadores iguales a cero.
    df = df.dropna(
        subset=["ndvi", "ndwi"]
    ).reset_index(drop=True)

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

def generate_tabular_dataset() -> pd.DataFrame:
    """Procesa las fechas oficiales de ambos lagos.

    Para cada lago y fecha:
    - descarga las bandas si no existen;
    - construye el DataFrame limpio;
    - guarda el CSV individual;
    - acumula las observaciones en un DataFrame global.

    Returns:
        DataFrame con todas las observaciones validas.
    """
    dataframes = []

    for lake in config.LAKES:
        print("\n" + "=" * 60)
        print(f"TABULAR: {lake.upper()}")
        print("=" * 60)

        for date in config.DATES[lake]:
            print(f"\n[process] {lake} {date}")

            df = process_date(
                lake,
                date,
            )

            print(
                f"[ok] {lake} {date}: "
                f"{len(df):,} observaciones validas"
            )

            dataframes.append(df)

    if not dataframes:
        raise ValueError(
            "no se generaron observaciones tabulares"
        )

    return pd.concat(
        dataframes,
        ignore_index=True,
    )

def print_dataset_summary(df: pd.DataFrame) -> None:
    """Muestra las estadisticas requeridas por el inciso 1.4."""

    print("\n" + "=" * 60)
    print("RESUMEN DEL DATASET")
    print("=" * 60)

    # Total
    print(
        f"\nTotal de observaciones: {len(df):,}"
    )

    # Por lago
    print("\nObservaciones por lago:")
    print(
        df.groupby("lake").size()
    )

    # Por lago y fecha
    print("\nObservaciones por lago y fecha:")
    print(
        df.groupby(["lake", "date"]).size()
    )

    # Variables
    print("\nVariables disponibles:")
    for column in df.columns:
        print(f"- {column}")

    # Tipos
    print("\nTipos de variables:")
    print(df.dtypes)

    # Missing values
    print("\nPorcentaje de valores faltantes:")

    missing_percentage = (
        df.isna().mean() * 100
    )

    print(
        missing_percentage.round(4)
    )

if __name__ == "__main__":
    df = generate_tabular_dataset()
    print_dataset_summary(df)