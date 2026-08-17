"""Análisis temporal y visualización (ejercicio 4).

Este módulo:
- lee lagos y fechas desde config.py (única fuente de verdad, no duplica
  listas ni rutas);
- lee data/rasters/{lake}/{date}/cyano.tif generados por indices.py,
  ignorando NoData/NaN (no reimplementa el cálculo del índice, ni toca
  Sentinel);
- calcula cyano_mean y valid_pixels por lago y fecha;
- genera outputs/tables/temporal_summary.csv con las columnas exactas del
  contrato (lake,date,cyano_mean,valid_pixels);
- genera outputs/figures/temporal_{lake}.png;
- identifica picos de floración y fechas críticas;
- redacta una interpretación breve basada únicamente en los resultados
  observados.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

import config

CYANO_FILENAME = "cyano.tif"

# Umbral solo para *advertir* en la interpretación que una fecha tiene poca
# cobertura válida (no filtra ni descarta datos). El cyano.tif ya trae una
# mascara de agua (waterBody() en indices.py), asi que valid_pixels/total
# ronda ~20-25% en TODAS las fechas de forma normal (el resto es tierra
# alrededor del lago dentro del bbox rectangular) -- eso no es una alerta.
# Lo que sí importa es cuánta agua del lago quedó cubierta en cada fecha
# respecto a las demás fechas del mismo lago (nubes, bordes de escena,
# etc.), así que el umbral se aplica sobre esa cobertura relativa. 0.75
# deja pasar el caso ya documentado en el pdf (Amatitlán 2026-02-07, ~57.1%
# de cobertura válida respecto al resto de fechas de ese lago).
LOW_COVERAGE_WARNING_RATIO = 0.75


def read_cyano_mean(lake: str, date: str) -> dict:
    """Lee cyano.tif de una fecha/lago y calcula su media ignorando NaN/NoData.

    No lanza excepción si el raster no existe todavía (por ejemplo si Persona
    2 aún no ha generado esa fecha); en ese caso regresa cyano_mean=None para
    que el resto del pipeline lo pueda saltar sin romperse.
    """
    path = config.raster_dir(lake, date) / CYANO_FILENAME
    if not path.exists():
        print(f"[warn] no existe {path}, se omite esta fecha")
        return {
            "lake": lake,
            "date": date,
            "cyano_mean": None,
            "valid_pixels": 0,
            "total_pixels": 0,
        }

    with rasterio.open(path) as src:
        data = src.read(1).astype("float64", copy=False)
        nodata = src.nodata

    valid_mask = np.isfinite(data)
    if nodata is not None:
        valid_mask &= data != nodata

    valid_pixels = int(valid_mask.sum())
    total_pixels = int(data.size)
    cyano_mean = float(data[valid_mask].mean()) if valid_pixels > 0 else None

    return {
        "lake": lake,
        "date": date,
        "cyano_mean": cyano_mean,
        "valid_pixels": valid_pixels,
        "total_pixels": total_pixels,
    }


def build_temporal_summary() -> list[dict]:
    """Recorre ambos lagos y sus fechas oficiales (config.DATES) y arma la
    lista de registros, uno por lago/fecha, ordenada cronológicamente.
    """
    records = [
        read_cyano_mean(lake, date)
        for lake in config.LAKES
        for date in config.DATES[lake]
    ]
    records.sort(key=lambda r: (r["lake"], r["date"]))
    return records


def save_temporal_summary_csv(records: list[dict], output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/temporal_summary.csv con exactamente las columnas
    lake,date,cyano_mean,valid_pixels (contrato del repo).
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.TABLES_DIR / "temporal_summary.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(config.TEMPORAL_SUMMARY_COLUMNS)
        for r in records:
            mean_str = "" if r["cyano_mean"] is None else f"{r['cyano_mean']:.6f}"
            writer.writerow([r["lake"], r["date"], mean_str, r["valid_pixels"]])

    print(f"[ok] tabla temporal guardada en {output_path}")
    return output_path


def plot_temporal_series(records: list[dict], lake: str, output_path: Path | None = None) -> Path:
    """Genera y guarda el gráfico de línea de cyano_mean vs fecha para un lago,
    marcando el pico máximo observado.
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.FIGURES_DIR / f"temporal_{lake}.png")

    lake_records = [r for r in records if r["lake"] == lake and r["cyano_mean"] is not None]
    dates = [r["date"] for r in lake_records]
    values = [r["cyano_mean"] for r in lake_records]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, values, marker="o", color="#2a7f62", linewidth=2)

    if values:
        peak_idx = int(np.argmax(values))
        ax.scatter(
            [dates[peak_idx]], [values[peak_idx]],
            color="crimson", zorder=5, s=80,
            label=f"pico: {dates[peak_idx]} ({values[peak_idx]:.2f})",
        )
        ax.legend()

    ax.set_title(f"Índice promedio de cianobacteria por fecha — {lake.capitalize()}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Índice de cianobacteria (promedio)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[ok] gráfica guardada en {output_path}")
    return output_path


def identify_peaks(records: list[dict], lake: str) -> dict:
    """Calcula máximo, mínimo y sus fechas para un lago, a partir de las
    fechas con cyano_mean válido.
    """
    lake_records = [r for r in records if r["lake"] == lake and r["cyano_mean"] is not None]
    if not lake_records:
        return {"lake": lake, "max_value": None, "min_value": None, "n_dates": 0}

    max_r = max(lake_records, key=lambda r: r["cyano_mean"])
    min_r = min(lake_records, key=lambda r: r["cyano_mean"])

    return {
        "lake": lake,
        "max_value": max_r["cyano_mean"],
        "max_date": max_r["date"],
        "min_value": min_r["cyano_mean"],
        "min_date": min_r["date"],
        "mean_of_means": float(np.mean([r["cyano_mean"] for r in lake_records])),
        "n_dates": len(lake_records),
    }


def build_interpretation(records: list[dict]) -> str:
    """Redacta una interpretación breve basada únicamente en los valores
    calculados (rango, promedio, fecha del pico) más una nota de cobertura
    para fechas con muchos píxeles inválidos.
    """
    lines = []
    for lake in config.LAKES:
        stats = identify_peaks(records, lake)
        if stats["max_value"] is None:
            lines.append(f"- {lake.capitalize()}: no hay datos válidos para interpretar todavía.")
            continue
        lines.append(
            f"- {lake.capitalize()}: el índice promedio de cianobacteria varió entre "
            f"{stats['min_value']:.2f} ({stats['min_date']}) y {stats['max_value']:.2f} "
            f"({stats['max_date']}), con un promedio de {stats['mean_of_means']:.2f} a lo "
            f"largo de {stats['n_dates']} fechas."
        )

    # Cobertura relativa: se compara valid_pixels de cada fecha contra el
    # máximo de valid_pixels visto en ese mismo lago (referencia de "agua
    # del lago totalmente cubierta"), no contra el total del raster.
    for lake in config.LAKES:
        lake_records = [r for r in records if r["lake"] == lake and r["valid_pixels"] > 0]
        if not lake_records:
            continue
        reference = max(r["valid_pixels"] for r in lake_records)
        if reference == 0:
            continue
        for r in lake_records:
            ratio = r["valid_pixels"] / reference
            if ratio < LOW_COVERAGE_WARNING_RATIO:
                lines.append(
                    f"- Nota: {r['lake'].capitalize()} {r['date']} tiene cobertura válida "
                    f"parcial (~{ratio * 100:.1f}% respecto al resto de fechas de ese lago); "
                    f"su promedio debe interpretarse con cautela."
                )

    return "\n".join(lines)


def verify_temporal_completeness(records: list[dict]) -> bool:
    """Verifica que el resumen temporal tenga las 22 filas esperadas (11 por
    lago). Solo imprime el estado -- no lanza excepción, para no tumbar el
    resto del análisis si a alguien todavía le falta una fecha.
    """
    total = len(records)
    ok = total == len(config.LAKES) * 11
    print(f"[check] filas totales en el resumen temporal: {total} (esperado: 22)")
    for lake in config.LAKES:
        n = sum(1 for r in records if r["lake"] == lake)
        status = "ok" if n == 11 else "FALTAN FECHAS"
        print(f"[check] {lake}: {n}/11 observaciones -> {status}")
        ok = ok and n == 11
    return ok


def identify_local_peaks(records: list[dict], lake: str) -> list[dict]:
    """Identifica máximos locales: fechas donde cyano_mean es mayor que la
    fecha anterior Y la fecha siguiente de ese mismo lago. Solo se evalúan
    puntos interiores (no el primero ni el último, que no tienen ambos
    vecinos). Sin SciPy, comparación directa punto a punto.
    """
    lake_records = [r for r in records if r["lake"] == lake and r["cyano_mean"] is not None]
    peaks = []
    for i in range(1, len(lake_records) - 1):
        prev_v = lake_records[i - 1]["cyano_mean"]
        curr = lake_records[i]
        next_v = lake_records[i + 1]["cyano_mean"]
        if curr["cyano_mean"] > prev_v and curr["cyano_mean"] > next_v:
            peaks.append({"date": curr["date"], "value": curr["cyano_mean"]})
    return peaks


def build_peaks_table(records: list[dict]) -> list[dict]:
    """Arma la tabla de máximo global, mínimo global y picos locales por
    lago, con sus fechas correspondientes (para temporal_peaks.csv).
    """
    rows = []
    for lake in config.LAKES:
        stats = identify_peaks(records, lake)
        if stats["max_value"] is not None:
            rows.append({"lake": lake, "tipo": "maximo_global", "date": stats["max_date"], "value": stats["max_value"]})
            rows.append({"lake": lake, "tipo": "minimo_global", "date": stats["min_date"], "value": stats["min_value"]})
        for p in identify_local_peaks(records, lake):
            rows.append({"lake": lake, "tipo": "pico_local", "date": p["date"], "value": p["value"]})
    return rows


def save_peaks_csv(rows: list[dict], output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/temporal_peaks.csv con columnas
    lake,tipo,date,value.
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.TABLES_DIR / "temporal_peaks.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lake", "tipo", "date", "value"])
        for r in rows:
            writer.writerow([r["lake"], r["tipo"], r["date"], f"{r['value']:.6f}"])

    print(f"[ok] tabla de picos guardada en {output_path}")
    return output_path


def build_lake_comparison(records: list[dict]) -> list[dict]:
    """Arma la tabla comparativa del ejercicio 7: métrica por lago
    (promedio, máximo, fecha del máximo, mínimo, cantidad de picos).
    """
    stats = {lake: identify_peaks(records, lake) for lake in config.LAKES}
    n_peaks = {lake: len(identify_local_peaks(records, lake)) for lake in config.LAKES}

    def fmt(v):
        return "" if v is None else f"{v:.4f}"

    rows = [
        {"metric": "promedio_cyano", **{lake: fmt(stats[lake].get("mean_of_means")) for lake in config.LAKES}},
        {"metric": "maximo_cyano", **{lake: fmt(stats[lake].get("max_value")) for lake in config.LAKES}},
        {"metric": "fecha_maximo", **{lake: stats[lake].get("max_date") or "" for lake in config.LAKES}},
        {"metric": "minimo_cyano", **{lake: fmt(stats[lake].get("min_value")) for lake in config.LAKES}},
        {"metric": "cantidad_picos", **{lake: str(n_peaks[lake]) for lake in config.LAKES}},
    ]
    return rows


def save_lake_comparison_csv(rows: list[dict], output_path: Path | None = None) -> Path:
    """Guarda outputs/tables/lake_comparison.csv con columnas
    metric,atitlan,amatitlan (una fila por métrica).
    """
    config.ensure_output_dirs()
    output_path = output_path or (config.TABLES_DIR / "lake_comparison.csv")
    fieldnames = ["metric"] + list(config.LAKES.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] tabla comparativa de lagos guardada en {output_path}")
    return output_path


def build_lake_comparison_interpretation(records: list[dict]) -> str:
    """Redacta la sección del ejercicio 7: compara intensidad, frecuencia y
    evolución temporal entre ambos lagos a partir de los resultados
    calculados, y separa eso de las posibles explicaciones ambientales
    (que no se derivan directamente de los datos de este laboratorio).
    """
    stats = {lake: identify_peaks(records, lake) for lake in config.LAKES}
    n_peaks = {lake: len(identify_local_peaks(records, lake)) for lake in config.LAKES}
    lakes = list(config.LAKES.keys())

    lines = ["## Ejercicio 7: análisis y comparación entre lagos\n"]
    lines.append("### Resultados observados\n")
    for lake in lakes:
        s = stats[lake]
        if s["max_value"] is None:
            lines.append(f"- **{lake.capitalize()}**: aún no hay datos suficientes para comparar.")
            continue
        lines.append(
            f"- **{lake.capitalize()}**: promedio de {s['mean_of_means']:.2f}, máximo de "
            f"{s['max_value']:.2f} el {s['max_date']}, mínimo de {s['min_value']:.2f} el "
            f"{s['min_date']}. Se identificaron {n_peaks[lake]} pico(s) local(es) (fechas más "
            f"altas que la anterior y la siguiente) en el período estudiado."
        )

    if all(stats[lake]["max_value"] is not None for lake in lakes):
        a, b = lakes[0], lakes[1]
        more_intense = a if stats[a]["mean_of_means"] > stats[b]["mean_of_means"] else b
        less_intense = b if more_intense == a else a

        lines.append("\n### Comparación directa\n")
        lines.append(
            f"- **Intensidad**: {more_intense.capitalize()} tuvo, en promedio, un índice de "
            f"cianobacteria más alto durante el período estudiado "
            f"({stats[more_intense]['mean_of_means']:.2f} vs "
            f"{stats[less_intense]['mean_of_means']:.2f})."
        )
        if n_peaks[a] != n_peaks[b]:
            more_frequent = a if n_peaks[a] > n_peaks[b] else b
            less_frequent = b if more_frequent == a else a
            lines.append(
                f"- **Frecuencia**: {more_frequent.capitalize()} mostró más picos locales "
                f"({n_peaks[more_frequent]} vs {n_peaks[less_frequent]}) en sus 11 fechas."
            )
        else:
            lines.append(
                f"- **Frecuencia**: ambos lagos mostraron la misma cantidad de picos locales "
                f"({n_peaks[a]}) en sus respectivas 11 fechas."
            )
        lines.append(
            "- **Evolución temporal y fechas críticas**: ver `temporal_atitlan.png` / "
            "`temporal_amatitlan.png` y `temporal_peaks.csv` para el detalle de cuándo ocurrió "
            "cada pico en cada lago."
        )
        lines.append(
            "- Para complementar con la distribución espacial dentro de cada lago (ejercicio 5), "
            "ver `comparativo_atitlan.png` / `comparativo_amatitlan.png` (generados por "
            "`src/spatial.py`)."
        )

    lines.append("\n### Posibles diferencias ambientales (hipótesis, no verificadas por este análisis)\n")
    lines.append(
        "Lo siguiente son explicaciones plausibles a partir de características generales "
        "conocidas de ambos lagos -- no conclusiones que se deriven estadísticamente de los "
        "datos de este laboratorio, y deben presentarse como tal en el informe:"
    )
    lines.append(
        "- **Geografía y profundidad**: Atitlán es un lago volcánico profundo, lo que favorece "
        "mayor estratificación térmica y dilución; Amatitlán es considerablemente más somero, lo "
        "que facilita el calentamiento superficial y la resuspensión de nutrientes."
    )
    lines.append(
        "- **Presión urbana y uso del suelo**: la cuenca de Amatitlán recibe drenaje de una zona "
        "densamente urbanizada (incluye parte del área metropolitana de Guatemala), con aportes "
        "históricos de aguas residuales y agrícolas; la cuenca de Atitlán tiene menor densidad "
        "urbana directa sobre el lago, aunque también recibe presión agrícola y de aguas "
        "residuales de los municipios aledaños."
    )
    lines.append(
        "- **Clima y temperatura**: aguas más cálidas -más probables en un cuerpo de agua somero "
        "como Amatitlán- favorecen la proliferación de cianobacterias."
    )
    lines.append(
        "- Estas explicaciones son hipótesis razonables a validar con datos adicionales "
        "(temperatura del agua, nutrientes, uso del suelo de la cuenca), no hallazgos "
        "confirmados por este laboratorio."
    )

    return "\n".join(lines)


def run_temporal_analysis() -> dict:
    """Punto de entrada único de este módulo: regenera automáticamente la
    tabla temporal y ambas gráficas a partir de los raster existentes en
    data/rasters/, cierra el ejercicio 4 (picos locales) y arma la
    comparación entre lagos del ejercicio 7.
    """
    records = build_temporal_summary()
    verify_temporal_completeness(records)

    csv_path = save_temporal_summary_csv(records)
    figure_paths = {lake: plot_temporal_series(records, lake) for lake in config.LAKES}
    peaks = {lake: identify_peaks(records, lake) for lake in config.LAKES}
    interpretation = build_interpretation(records)

    print("\n=== Interpretación inicial (ejercicio 4) ===")
    print(interpretation)

    # ejercicio 4: cierre (picos locales + tabla de picos)
    peaks_rows = build_peaks_table(records)
    peaks_csv_path = save_peaks_csv(peaks_rows)

    # ejercicio 7: comparación entre lagos
    comparison_rows = build_lake_comparison(records)
    comparison_csv_path = save_lake_comparison_csv(comparison_rows)
    comparison_interpretation = build_lake_comparison_interpretation(records)

    print("\n" + comparison_interpretation)

    return {
        "records": records,
        "csv_path": csv_path,
        "figure_paths": figure_paths,
        "peaks": peaks,
        "interpretation": interpretation,
        "peaks_rows": peaks_rows,
        "peaks_csv_path": peaks_csv_path,
        "comparison_rows": comparison_rows,
        "comparison_csv_path": comparison_csv_path,
        "comparison_interpretation": comparison_interpretation,
    }


if __name__ == "__main__":
    run_temporal_analysis()