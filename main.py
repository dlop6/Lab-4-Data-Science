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

    # paso 2: indices (persona 2)
    from src.indices import generate_indices
    generate_indices()

    # paso 3: analisis temporal (persona 3)
    from src.temporal import run_temporal_analysis
    run_temporal_analysis()

    # paso 4: analisis espacial / ejercicio 5 (persona 1)
    from src.spatial import generate_all_spatial_maps
    generate_all_spatial_maps()

    # paso 5: correlaciones / ejercicio 6 (persona 2)
    from src.correlation import run_correlation_analysis
    run_correlation_analysis()

    # paso 6: distribuciones / ejercicio 8.3 (persona 2)
    from src.distributions import run_distribution_analysis
    run_distribution_analysis()

    # paso 7: comparacion entre lagos / ejercicio 7, cierre ejercicio 4 (persona 3)
    # (ya se ejecuto en el paso 3 via run_temporal_analysis(), ver arriba)

    # paso 8: analisis exploratorio adicional / ejercicio 8.1, 8.2, 8.4, 8.5 (persona 3)
    from src.exploratory import run_exploratory_analysis
    run_exploratory_analysis()

    print("listo. corre 'python scripts/validar_descarga.py' para probar la conexion a sentinel hub.")


if __name__ == "__main__":
    main()
