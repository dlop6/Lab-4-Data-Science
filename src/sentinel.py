"""
acceso a sentinel hub. toda la autenticacion y el request generico viven aca
para que nadie mas tenga que reimplementar esto.

persona 2 solo debe importar request_product() y pasarle su propio evalscript
(ndvi, ndwi o el cyano detection script), sin tocar nada de este archivo.
"""

from pathlib import Path

from dotenv import load_dotenv
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
)
from sentinelhub.geo_utils import bbox_to_dimensions

import config

load_dotenv()  # carga SH_CLIENT_ID / SH_CLIENT_SECRET desde .env si existe

# resolucion en metros/pixel usada para todas las descargas. 10m alcanza para
# ndvi/ndwi/cyano con las bandas nativas de sentinel-2 (b02,b03,b04,b08).
RESOLUTION = 10

# la process api de sentinel hub no acepta mas de 2500px por lado. atitlan es
# lo bastante grande (~25km) para pasarse de eso a 10m/pixel, asi que hay que
# limitar el tamano y dejar que la resolucion efectiva suba un poco en vez de
# que el request falle.
MAX_SIZE_PX = 2500

# el proyecto usa cuentas del Copernicus Data Space Ecosystem (CDSE), no el
# sentinel-hub.com comercial clasico, asi que hay que apuntar explicitamente
# a sus endpoints de auth/servicio.
CDSE_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# coleccion de sentinel-2 L2A servida desde CDSE (con DataCollection.SENTINEL2_L2A
# a secas apunta al servicio comercial y falla con credenciales de CDSE)
CDSE_SENTINEL2_L2A = DataCollection.SENTINEL2_L2A.define_from(
    name="s2l2a_cdse", service_url=CDSE_BASE_URL
)
CDSE_SENTINEL2_L1C = DataCollection.SENTINEL2_L1C.define_from(
    name="s2l1c_cdse", service_url=CDSE_BASE_URL
)


def _build_config() -> SHConfig:
    """arma el SHConfig leyendo credenciales de variables de entorno.

    las credenciales nunca quedan escritas en el codigo ni en el repo, solo
    se leen del entorno (.env local, que esta en .gitignore).
    """
    sh_config = SHConfig()
    sh_config.sh_client_id = _get_env("SH_CLIENT_ID")
    sh_config.sh_client_secret = _get_env("SH_CLIENT_SECRET")
    sh_config.sh_base_url = CDSE_BASE_URL
    sh_config.sh_token_url = CDSE_TOKEN_URL
    return sh_config


def _get_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value or "tu_client_id_aqui" in value or "tu_client_secret_aqui" in value:
        raise EnvironmentError(
            f"falta configurar {name}. copia .env.example a .env y pon tus "
            f"credenciales reales de https://shapps.dataspace.copernicus.eu/dashboard/"
        )
    return value


def _lake_bbox(lake: str) -> BBox:
    """arma el BBox de sentinelhub a partir de las coordenadas de config.py."""
    if lake not in config.LAKES:
        raise ValueError(f"lago desconocido: {lake}. debe ser uno de {list(config.LAKES)}")
    coords = config.LAKES[lake]
    return BBox(
        bbox=[coords["west"], coords["south"], coords["east"], coords["north"]],
        crs=CRS.WGS84,
    )


def _clamp_size(size: tuple) -> tuple:
    """si el tamano calculado se pasa del limite de la process api (2500px por
    lado), lo reduce proporcionalmente manteniendo el aspect ratio del bbox.
    """
    width, height = size
    largest = max(width, height)
    if largest <= MAX_SIZE_PX:
        return size
    scale = MAX_SIZE_PX / largest
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def request_product(lake: str, date: str, evalscript: str, output_path: Path, data_collection=CDSE_SENTINEL2_L2A) -> Path:
    """pide a sentinel hub el producto de una fecha/lago usando el evalscript dado,
    guarda el resultado (tif) en output_path y regresa su Path.

    esta es la unica funcion que debe usarse para pedirle algo a sentinel hub
    en todo el proyecto. no reimplementar autenticacion ni bbox en otro lado.

    args:
        lake: "atitlan" o "amatitlan" (debe existir en config.LAKES)
        date: fecha en formato YYYY-DD, debe estar en config.DATES[lake]
        evalscript: el evalscript de sentinel hub (ndvi, ndwi, cyano, etc.)
        output_path: ruta donde se va a guardar el .tif resultante

    returns:
        Path del archivo guardado (igual a output_path)
    """
    if date not in config.DATES.get(lake, []):
        raise ValueError(f"fecha {date} no esta en las fechas oficiales de {lake}")

    sh_config = _build_config()
    bbox = _lake_bbox(lake)
    # bbox_to_dimensions calcula el ancho/alto en pixeles a partir del bbox
    # (en grados, WGS84) y la resolucion real en metros. pasar resolution
    # directo a SentinelHubRequest con un bbox en grados lo interpreta como
    # "grados por pixel" y revienta con bboxes gigantes o diminutos.
    size = bbox_to_dimensions(bbox, resolution=RESOLUTION)
    size = _clamp_size(size)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=(date, date),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=sh_config,
        data_folder=str(output_path.parent),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = request.get_data(save_data=False)

    # get_data regresa una lista de arrays (uno por response definida arriba)
    _save_tiff(data[0], bbox, output_path)
    return output_path


def _save_tiff(array, bbox: BBox, output_path: Path) -> None:
    """guarda un array numpy como geotiff georreferenciado con rasterio."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    else:
        arr = np.moveaxis(arr, -1, 0)  # (h, w, bands) -> (bands, h, w)

    count, height, width = arr.shape
    transform = from_bounds(*bbox, width=width, height=height)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=arr.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(arr)
