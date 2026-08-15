"""Generacion de NDVI, NDWI e indice de cianobacteria

Este modulo:
- importa lagos, fechas y rutas desde config.py;
- reutiliza request_product() de src.sentinel;
- guarda solo data/rasters/{lake}/{date}/{ndvi,ndwi,cyano}.tif;
- usa NaN para pixeles sin datos/no validos en productos FLOAT32.
"""

from pathlib import Path

import numpy as np
import rasterio

import config
from src.sentinel import (request_product, CDSE_SENTINEL2_L1C)


NDVI_EVALSCRIPT = r"""
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "dataMask"] }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [NaN];
  let den = sample.B08 + sample.B04;
  if (den === 0) return [NaN];
  return [(sample.B08 - sample.B04) / den];
}
"""


NDWI_EVALSCRIPT = r"""
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B03", "B08", "dataMask"] }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  if (sample.dataMask === 0) return [NaN];
  let den = sample.B03 + sample.B08;
  if (den === 0) return [NaN];
  return [(sample.B03 - sample.B08) / den];
}
"""


# Adaptacion numerica del script oficial "Cyanobacteria Chlorophyll-a NDCI L1C"
# de Sentinel Hub. Conserva su mascara de agua y su ecuacion de chlorophyll-a,
# pero devuelve un raster FLOAT32 de concentracion en vez de colores RGB.
CYANO_EVALSCRIPT = r"""
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B07", "B08", "B8A", "B11", "B12", "dataMask"]
    }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}

var MNDWI_threshold = 0.42;
var NDWI_threshold = 0.4;
var filter_UABS = true;

function safeRatio(num, den) {
  return den === 0 ? NaN : num / den;
}

function waterBody(r, g, b, nir, swir1, swir2) {
  let ndvi = safeRatio(nir-r, nir+r);
  let mndwi = safeRatio(g-swir1, g+swir1);
  let ndwi = safeRatio(g-nir, g+nir);
  let ndwi_leaves = safeRatio(nir-swir1, nir+swir1);
  let aweish = b + 2.5*g - 1.5*(nir+swir1) - 0.25*swir2;
  let aweinsh = 4*(g-swir1) - (0.25*nir + 2.75*swir1);
  let dbsi = safeRatio(swir1-g, swir1+g) - ndvi;

  let ws = 0;
  if (mndwi > MNDWI_threshold || ndwi > NDWI_threshold ||
      aweinsh > 0.1879 || aweish > 0.1112 || ndvi < -0.2 ||
      ndwi_leaves > 1) {
    ws = 1;
  }

  if (filter_UABS && ws === 1 && (aweinsh <= -0.03 || dbsi > 0)) {
    ws = 0;
  }
  return ws;
}

function evaluatePixel(s) {
  if (s.dataMask === 0) return [NaN];
  if (waterBody(s.B04, s.B03, s.B02, s.B08, s.B11, s.B12) === 0) return [NaN];

  let den = s.B05 + s.B04;
  if (den === 0) return [NaN];

  let ndci = (s.B05 - s.B04) / den;
  // Ecuacion usada por el evalscript oficial de CyanoLakes/Sentinel Hub.
  let chl = 826.57*Math.pow(ndci, 3) - 176.43*Math.pow(ndci, 2) + 19*ndci + 4.071;

  if (!isFinite(chl) || chl < 0) return [NaN];
  return [chl]; 
}
"""


PRODUCTS = {
    "ndvi.tif": NDVI_EVALSCRIPT,
    "ndwi.tif": NDWI_EVALSCRIPT,
    "cyano.tif": CYANO_EVALSCRIPT,
}


def validate_raster(path: Path) -> dict:
    """Abre un GeoTIFF y comprueba que sea utilizable y tenga pixeles validos."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no existe el raster esperado: {path}")

    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"{path} debe tener 1 banda; tiene {src.count}")
        data = src.read(1).astype("float64", copy=False)
        valid = np.isfinite(data)
        valid_pixels = int(valid.sum())
        if valid_pixels == 0:
            raise ValueError(f"{path} no contiene pixeles validos")
        return {
            "path": path,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "valid_pixels": valid_pixels,
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
        }


def generate_indices(skip_existing: bool = True, validate: bool = True) -> list[Path]:
    """Genera NDVI, NDWI y cianobacteria para ambos lagos y todas sus fechas.

    Args:
        skip_existing: si True, no vuelve a descargar un producto que ya existe.
        validate: si True, abre cada raster al terminar y verifica que tenga datos.

    Returns:
        Lista de rutas de los productos generados/encontrados.
    """
    config.ensure_output_dirs()
    outputs: list[Path] = []

    for lake in config.LAKES:
        for date in config.DATES[lake]:
            out_dir = config.raster_dir(lake, date)
            out_dir.mkdir(parents=True, exist_ok=True)

            for filename, evalscript in PRODUCTS.items():
                output_path = out_dir / filename

                if skip_existing and output_path.exists():
                    print(f"[skip] {lake} {date} {filename}")
                else:
                    print(f"[request] {lake} {date} {filename}")
                    if filename == "cyano.tif":
                      request_product(lake, date, evalscript, output_path, data_collection=CDSE_SENTINEL2_L1C)
                    else:
                      request_product(lake, date, evalscript, output_path)


                if validate:
                    info = validate_raster(output_path)
                    print(
                        f"[ok] {output_path} | {info['width']}x{info['height']} | "
                        f"validos={info['valid_pixels']} | rango={info['min']:.4f}..{info['max']:.4f}"
                    )
                outputs.append(output_path)

    return outputs
