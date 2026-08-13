"""
punto de entrada del proyecto. orquesta los 3 pasos del laboratorio:
descarga -> indices -> analisis temporal.

por ahora solo esta el esqueleto; persona 2 y persona 3 van a completar
generate_indices() y run_temporal_analysis() cuando implementen sus modulos.
"""

import config


def main():
    config.ensure_output_dirs()

    # paso 1: descarga (persona 1) - request_product() ya esta listo en src/sentinel.py
    print("estructura de datos verificada.")

    # paso 2: indices (persona 2) - pendiente de implementar
    # from src.indices import generate_indices
    # generate_indices()

    # paso 3: analisis temporal (persona 3) - pendiente de implementar
    # from src.temporal import run_temporal_analysis
    # run_temporal_analysis()

    print("listo. corre 'python scripts/validar_descarga.py' para probar la conexion a sentinel hub.")


if __name__ == "__main__":
    main()
