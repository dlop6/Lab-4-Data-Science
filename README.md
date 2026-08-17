# Laboratorio 4 - Análisis de Datos Geoespaciales

Análisis de floraciones de cianobacteria en los lagos Atitlán y Amatitlán mediante imágenes Sentinel-2. Proyecto de CC3084 - Data Science, Universidad del Valle de Guatemala.

## Informe

El análisis, las figuras, las tablas y las conclusiones se encuentran en el [informe final](./informe_final_lab4.pdf).

## Preparación

```bash
python -m venv .venv
pip install -r requirements.txt
```

Activa el entorno virtual y copia `.env.example` como `.env`. Luego configura allí las credenciales de Sentinel Hub.

## Ejecución

```bash
python main.py
```

El comando ejecuta el flujo completo y guarda los resultados en `outputs/`. Para comprobar únicamente la conexión y una descarga de prueba:

```bash
python scripts/validar_descarga.py
```

## Estructura

```text
config.py        configuración central: lagos, fechas y rutas
main.py          punto de entrada del análisis
src/             obtención, procesamiento y análisis
data/geojson/    contornos de los lagos
data/rasters/    productos NDVI, NDWI y cianobacteria
outputs/         tablas y figuras generadas
```

Las fechas oficiales, nombres de los lagos y rutas se definen únicamente en `config.py`.
