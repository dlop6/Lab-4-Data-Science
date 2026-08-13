# Lab 4 — Análisis de Datos Geoespaciales

Monitoreo de floraciones de cianobacteria en los lagos Atitlán y Amatitlán usando Sentinel-2 / Sentinel Hub. CC3084 - Data Science, Semestre II 2026.

## Instalación

```bash
python -m venv .venv
# windows
.venv\Scripts\activate
# linux/mac
source .venv/bin/activate

pip install -r requirements.txt
```

Copia `.env.example` a `.env` y coloca tus credenciales de Sentinel Hub (client id / secret, se crean en https://apps.sentinel-hub.com):

```bash
cp .env.example .env
```

## Ejecución

```bash
# valida que la conexión con sentinel hub funciona (requiere .env configurado)
python scripts/validar_descarga.py

# corre el pipeline completo
python main.py
```

## Estructura

```
config.py          coordenadas, fechas oficiales y rutas centralizadas
src/sentinel.py     autenticación y descarga (request_product())
src/indices.py      NDVI, NDWI, cianobacteria (persona 2)
src/temporal.py     análisis temporal (persona 3)
data/geojson/       contorno de cada lago
data/rasters/       rasters generados por fecha/lago (no se versionan, ver .gitignore)
outputs/            tabla y gráficas del análisis temporal
```

## Convenciones (no modificar sin acuerdo del grupo)

- Lagos: `atitlan`, `amatitlan`.
- Fechas: `YYYY-MM-DD`, únicamente las 11 oficiales por lago (`config.DATES`).
- Rasters: `ndvi.tif`, `ndwi.tif`, `cyano.tif` en `data/rasters/{lake}/{date}/`.
- CSV temporal: `outputs/tables/temporal_summary.csv` con columnas `lake,date,cyano_mean,valid_pixels`.
