"""
config central del proyecto. aca viven, una sola vez, las coordenadas de los
lagos, las fechas oficiales del laboratorio y todas las rutas base.

nadie mas deberia duplicar esta info en otro archivo (persona 2 y 3 importan
de aca directo). si algo cambia, cambia solo aca.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# rutas base del proyecto (todo relativo a este archivo, nada hardcodeado
# a una ruta local de una compu en especifico)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
GEOJSON_DIR = DATA_DIR / "geojson"
RASTERS_DIR = DATA_DIR / "rasters"

TABULAR_DIR = DATA_DIR / "tabular"

OUTPUTS_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SPATIAL_FIGURES_DIR = FIGURES_DIR / "spatial"

# ---------------------------------------------------------------------------
# lagos: coordenadas oficiales (bounding box) del pdf del laboratorio
# ---------------------------------------------------------------------------
LAKES = {
    "atitlan": {
        "west": -91.326256,
        "east": -91.07151,
        "south": 14.5948,
        "north": 14.750979,
    },
    "amatitlan": {
        "west": -90.638065,
        "east": -90.512924,
        "south": 14.412347,
        "north": 14.493799,
    },
}

# ruta al geojson de contorno real de cada lago (usado para recortar el area
# de interes en vez de usar solo el bbox rectangular)
GEOJSON_PATHS = {
    "atitlan": GEOJSON_DIR / "atitlan.geojson",
    "amatitlan": GEOJSON_DIR / "amatitlan.geojson",
}

# ---------------------------------------------------------------------------
# fechas oficiales por lago (unica fuente de verdad, exactamente las 11 de
# cada lago que vienen en el pdf). nadie mas debe reescribir estas listas.
# ---------------------------------------------------------------------------
DATES = {
    "atitlan": [
        "2025-01-18",
        "2025-04-13",
        "2025-05-13",
        "2025-07-17",
        "2025-11-21",
        "2025-12-29",
        "2026-02-12",
        "2026-03-24",
        "2026-04-13",
        "2026-04-28",
        "2026-07-22",
    ],
    "amatitlan": [
        "2025-01-28",
        "2025-04-15",
        "2025-04-28",
        "2025-11-24",
        "2026-01-08",
        "2026-02-02",
        "2026-02-07",  # nota del pdf: cobertura valida parcial (~57.1%)
        "2026-03-29",
        "2026-04-13",
        "2026-04-28",
        "2026-06-19",
    ],
}

# nombres de raster que produce cada fecha (convencion obligatoria del contrato)
RASTER_NAMES = ("ndvi.tif", "ndwi.tif", "cyano.tif")

# columnas exactas que debe tener el csv de resumen temporal (persona 3)
TEMPORAL_SUMMARY_COLUMNS = ("lake", "date", "cyano_mean", "valid_pixels")


def raster_dir(lake: str, date: str) -> Path:
    """ruta de la carpeta de rasters para un lago y fecha especifico."""
    if lake not in LAKES:
        raise ValueError(f"lago desconocido: {lake}. debe ser uno de {list(LAKES)}")
    if date not in DATES[lake]:
        raise ValueError(f"fecha {date} no esta en las fechas oficiales de {lake}")
    return RASTERS_DIR / lake / date


def spatial_figures_dir(lake: str) -> Path:
    """ruta de la carpeta de mapas espaciales (ejercicio 5) para un lago."""
    if lake not in LAKES:
        raise ValueError(f"lago desconocido: {lake}. debe ser uno de {list(LAKES)}")
    return SPATIAL_FIGURES_DIR / lake


def ensure_output_dirs() -> None:
    """crea las carpetas de salida si no existen (idempotente, no rompe nada)."""
    dirs = [GEOJSON_DIR, RASTERS_DIR, TABULAR_DIR, TABLES_DIR, FIGURES_DIR]
    dirs += [spatial_figures_dir(lake) for lake in LAKES]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
