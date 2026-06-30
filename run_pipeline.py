"""
run_pipeline.py — orquesta la generacion de mapas de Chl-a con la GNN.

Fases: [1] descarga Sentinel-2  ->  [2] correccion atmosferica C2RCC (SNAP)  ->
[3] prediccion con la GNN  ->  [4] GeoTIFFs  ->  [5] plots  ->  [6] GIF.

Uso:
    python3 run_pipeline.py --date 2022-07-14
"""

import argparse
import os
import csv
import json
import yaml
import subprocess
import time
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True)
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

requested_date = args.date
fecha = args.date
search_window_days = int(cfg.get("search_window_days", 0))
max_cloud_cover = float(cfg.get("max_cloud_cover", 100))
safe_dir = cfg.get("safe_dir")
snap_dir = cfg.get("snap_dir")
pred_dir = cfg.get("pred_dir")
map_dir = cfg.get("map_dir")
geojson_file = cfg.get("geojson_file")
colormap_file = cfg.get("colormap_file")
bathymetry_map = cfg.get("bathymetry_map")
gnn_model_dir = cfg.get("gnn", {}).get("model_dir", "/app/gnn_models")

t_start = time.time()

print(f"\n=== [1] Descargando producto para {requested_date} ===")
t1 = time.time()
# Limpiar marcador previo para no leer una fecha resuelta antigua.
resolved_marker = os.path.join(safe_dir, "resolved_date.txt")
try:
    os.remove(resolved_marker)
except OSError:
    pass
try:
    subprocess.run(
        ["python3", "fetch/productFetcher.py", "--date", requested_date,
         "--output", safe_dir, "--window", str(search_window_days),
         "--max-cloud", str(max_cloud_cover)],
        check=True,
    )
except subprocess.CalledProcessError:
    print("No se encontraron productos para esa fecha (ni cercanas). El pipeline se detendra.")
    sys.exit(1)

# El fetcher escribe la fecha realmente descargada (puede diferir si se uso la mas cercana).
if os.path.isfile(resolved_marker):
    with open(resolved_marker) as f:
        resolved = f.read().strip()
    if resolved and resolved != requested_date:
        print(f"[fecha] Solicitada {requested_date} sin producto; usando la mas cercana: {resolved}")
        fecha = resolved
    elif resolved:
        fecha = resolved

subprocess.run(["python3", "fetch/productFetcher_tozip.py", "--date", fecha, "--input", safe_dir], check=True)
t_download = time.time() - t1
print(f"Tiempo transcurrido [1]: {t_download:.2f} s")

print(f"\n=== [2] Aplicando correccion atmosferica con SNAP ===")
t3 = time.time()
subprocess.run(["bash", "fetch/snap_batch_application.sh", fecha, safe_dir, snap_dir], check=True)
t_snap = time.time() - t3
print(f"Tiempo transcurrido [2]: {t_snap:.2f} s")

print(f"\n=== [3] Ejecutando prediccion GNN ===")
t5 = time.time()
subprocess.run(
    ["python3", "models/Aplicacion_Modelos_GNN.py",
     "--date", fecha, "--input", snap_dir, "--models", gnn_model_dir,
     "--pred", pred_dir, "--geojson", geojson_file, "--config", args.config],
    check=True,
)
t_predict = time.time() - t5
print(f"Tiempo transcurrido [3]: {t_predict:.2f} s")

print(f"\n=== [4] Generando TIFFs ===")
t7 = time.time()
subprocess.run(["python3", "models/Aplicacion_TIFFfromCSV.py", "--date", fecha, "--input", pred_dir, "--output", map_dir], check=True)
t_tiff = time.time() - t7
print(f"Tiempo transcurrido [4]: {t_tiff:.2f} s")

t_plots = 0.0
if cfg.get("plot_individuales", False):
    print(f"\n=== [5] Generando plots individuales ===")
    t9 = time.time()
    subprocess.run(["python3", "models/Aplicacion_PlotTIFF.py", "--date", fecha, "--input", map_dir, "--output", map_dir, "--colormap", colormap_file, "--bathymetry", bathymetry_map], check=True)
    t_plots = time.time() - t9
    print(f"Tiempo transcurrido [5]: {t_plots:.2f} s")

t_gif = 0.0
if cfg.get("generate_gif", False):
    print(f"\n=== [6] Generando GIF ===")
    t11 = time.time()
    subprocess.run(["python3", "models/Aplicacion_GenerateGif.py", "--date", fecha, "--input", map_dir, "--output", map_dir, "--colormap", colormap_file, "--bathymetry", bathymetry_map], check=True)
    t_gif = time.time() - t11
    print(f"Tiempo transcurrido [6]: {t_gif:.2f} s")

t_total = time.time() - t_start

# --- Desglose de la fase [3] desde el sidecar (feature-engineering vs inferencia) ---
feateng_s = infer_s = float("nan")
n_pixels = ""
sidecar = os.path.join(pred_dir, f"{fecha}_timing.json")
if os.path.isfile(sidecar):
    try:
        with open(sidecar) as f:
            tj = json.load(f)
        feateng_s = tj.get("feateng_s", float("nan"))
        infer_s = tj.get("infer_s", float("nan"))
        n_pixels = tj.get("n_pixels", "")
    except (OSError, ValueError):
        pass

print("\nPipeline completado correctamente.")
print(f"Tiempo total del pipeline: {t_total:.2f} s ({t_total/60:.2f} min)")
print(f"TOTAL_TIME_S: {t_total:.2f}")            # lo captura webapp.py para la UI

# --- Validez del resultado --------------------------------------------------
# Si TODOS los mapas de profundidad son de un solo color (todos los pixeles
# iguales), la imagen Sentinel-2 de origen no tenia calidad suficiente para que
# el pipeline funcione bien: marcamos el resultado como no valido y avisamos.
def _depth_is_flat(tif_path, tol=1e-4):
    """True si el mapa no tiene variacion real (un solo color); None si no se lee."""
    try:
        import numpy as np
        import rasterio
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
    except Exception:                              # noqa: BLE001
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True                                # sin datos -> degenerado
    return float(np.ptp(finite)) <= tol

_flat = [
    _depth_is_flat(f"{map_dir}{fecha}_chl_map_{d}.tif")
    for d in ("0_1", "1_2", "2_3", "3_4")
]
_known = [c for c in _flat if c is not None]
# Valido si hay al menos un mapa con variacion real; si no se pudo leer ninguno,
# no penalizamos (se asume valido).
map_valid = any(c is False for c in _known) if _known else True
print(f"MAP_VALID: {str(map_valid).lower()}")     # lo captura webapp.py para la UI
if not map_valid:
    print("[aviso] Todos los mapas son de un solo color: la imagen Sentinel-2 de "
          "origen no tenia calidad suficiente. Prueba con otra fecha.")

# --- CSV unico con TODAS las fases (persistente) -------------------------------
timings_csv = cfg.get("timings_csv", os.path.join(map_dir, "inference_times.csv"))
CSV_HEADER = [
    "timestamp", "requested_date", "resolved_date", "n_pixels",
    "download_s", "snap_s", "feateng_s", "infer_s",
    "prediction_phase_s", "tiff_s", "plots_s", "gif_s", "total_s", "valid",
]
try:
    os.makedirs(os.path.dirname(timings_csv), exist_ok=True)

    # Migracion: si ya existe un CSV sin la columna "valid", se reescribe la
    # cabecera y se rellena la nueva columna como vacia en las filas previas.
    old_rows = None
    if os.path.isfile(timings_csv):
        with open(timings_csv, newline="") as f:
            existing = list(csv.reader(f))
        if existing and "valid" not in existing[0]:
            old_rows = existing[1:]

    def _f(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) and x == x else ""  # x==x => no NaN

    new_row = [
        datetime.now().isoformat(timespec="seconds"),
        requested_date, fecha, n_pixels,
        _f(t_download), _f(t_snap), _f(feateng_s), _f(infer_s),
        _f(t_predict), _f(t_tiff), _f(t_plots), _f(t_gif), _f(t_total),
        "TRUE" if map_valid else "FALSE",
    ]

    if old_rows is not None:
        # Reescribir con la cabecera nueva, rellenando "valid" vacio en lo antiguo.
        with open(timings_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            for r in old_rows:
                w.writerow(r + [""])
            w.writerow(new_row)
    else:
        write_header = not os.path.isfile(timings_csv)
        with open(timings_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(CSV_HEADER)
            w.writerow(new_row)
    print(f"Tiempos registrados en {timings_csv}")
except OSError as e:
    print(f"[aviso] no se pudo escribir el CSV de tiempos ({e})")
