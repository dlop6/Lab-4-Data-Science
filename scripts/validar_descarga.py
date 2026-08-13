"""
prueba de descarga funcional: pide 1 producto de atitlan y 1 de amatitlan
usando request_product() para validar que la conexion con sentinel hub
funciona de punta a punta.

requiere tener SH_CLIENT_ID y SH_CLIENT_SECRET reales en .env (ver
.env.example). si no estan configurados, este script lo indica claramente
y no truena con un traceback feo.

uso:
    python scripts/validar_descarga.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.sentinel import request_product

# evalscript minimo solo para validar la conexion (banda B04 cruda, no es
# ndvi/ndwi/cyano todavia, eso lo hace persona 2 con su propio evalscript)
EVALSCRIPT_VALIDACION = """
//VERSION=3
function setup() {
  return {
    input: ["B04"],
    output: { bands: 1, sampleType: "UINT16" }
  };
}
function evaluatePixel(sample) {
  return [sample.B04];
}
"""


def main():
    pruebas = [
        ("atitlan", config.DATES["atitlan"][0]),
        ("amatitlan", config.DATES["amatitlan"][0]),
    ]

    for lake, date in pruebas:
        output_path = config.RASTERS_DIR / lake / date / "validacion_b04.tif"
        print(f"pidiendo producto de validacion: {lake} / {date} ...")
        try:
            path = request_product(lake, date, EVALSCRIPT_VALIDACION, output_path)
            print(f"  ok -> {path}")
        except EnvironmentError as e:
            print(f"  no se pudo validar todavia: {e}")
            print("  configura .env con credenciales reales y vuelve a correr este script.")
            return
        except Exception as e:
            print(f"  fallo la descarga para {lake}/{date}: {e}")
            raise

    print("\nvalidacion completa: ambos lagos respondieron correctamente.")


if __name__ == "__main__":
    main()
