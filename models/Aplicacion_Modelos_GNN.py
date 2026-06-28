"""
Aplicacion_Modelos_GNN.py — paso de prediccion con la GNN.

Reemplaza al `Aplicacion_Modelos.py` tabular: el feature-engineering (extraccion
de pixeles, combinaciones de bandas, estacion) es identico, pero la prediccion la
hace la red neuronal de grafos via `gnn.inference.predict_map_gnn` en lugar de los
modelos tabulares.

Salida: `{pred_dir}/{date}_pred.csv` con columnas
    Date, Latitude, Longitude, Chl_pred_0_1, Chl_pred_1_2, Chl_pred_2_3, Chl_pred_3_4
(igual que el pipeline tabular, para que `Aplicacion_TIFFfromCSV.py` no cambie).

La seleccion depth -> (dataset_name, input_df) se lee de `config.yaml` (clave `gnn`),
de modo que apunte a los checkpoints GNN entrenados que se monten en `model_dir`.
"""

import os
import sys
import json
import time
import argparse

import pandas as pd
import yaml

# El paquete vendorizado `gnn/` esta en la raiz del repo (un nivel por encima de models/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Aplicacion_utils import (  # noqa: E402  (mismo dir que este script)
    extract_pixels_in_marmenor,
    create_processed_dfs,
    add_band_combinations,
    compactar_prefijos_columnas,
    add_season,
)
from gnn.inference import predict_map_gnn  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, type=str, help="Fecha (YYYY-MM-DD)")
parser.add_argument("--input", required=True, help="Directorio del .tif procesado por SNAP")
parser.add_argument("--models", required=True, help="Directorio con los checkpoints GNN")
parser.add_argument("--pred", required=True, help="Directorio de salida para el CSV de predicciones")
parser.add_argument("--geojson", required=True, help="GeoJSON del Mar Menor")
parser.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yaml"))
args = parser.parse_args()

with open(args.config, "r") as f:
    CFG = yaml.safe_load(f)
GNN_CFG = CFG.get("gnn", {})
MODEL_NAME = GNN_CFG.get("model_name", "GraphSAGE")
KNN_K = int(GNN_CFG.get("knn_k", 8))
DEVICE = GNN_CFG.get("device", "cpu")
SELECTION = GNN_CFG.get("selection", {})

folder_path = args.input
polygon_path = args.geojson
date_str = str(args.date)
target_dates = [date_str]

# --- 1. Extraccion de pixeles + reflectancias (identico al tabular) ----------
t_feat0 = time.perf_counter()
groupings = ["5x5"]
net_set = ["C2X-Complex"]
for grouping in groupings:
    for net in net_set:
        df_tiffs = extract_pixels_in_marmenor(folder_path, target_dates, grouping, net, polygon_path)
        df_tiffs["Date"] = pd.to_datetime(df_tiffs["Date"])
        df_tiffs.to_csv(f"{folder_path}/df_tifs_{net}_{grouping}_{target_dates[0]}.csv", index=False)
print("DataFrames con reflectancias cargados")

# --- 2. Carga de los CSV de tifs y limpieza ----------------------------------
dfs_tifs = {}
for archivo in os.listdir(folder_path):
    if archivo.startswith("df_tifs_") and archivo.endswith(f"{target_dates[0]}.csv") and "planet" not in archivo:
        nombre_sin_extension = os.path.splitext(archivo)[0]
        ruta_completa = os.path.join(folder_path, archivo)
        dfs_tifs[nombre_sin_extension[:-11]] = pd.read_csv(ruta_completa)

for nombre_df, df in dfs_tifs.items():
    dfs_tifs[nombre_df] = df.dropna()

# --- 3. Feature engineering (procesados + combinaciones + estacion) ----------
dfs_tifs_all = create_processed_dfs(dfs_tifs)
dfs = add_band_combinations(dfs_tifs_all)
for nombre_df, df in dfs.items():
    dfs[nombre_df] = compactar_prefijos_columnas(df)
dfs = add_season(dfs)
t_feateng = time.perf_counter() - t_feat0
print(f"Dataframes con combinaciones de bandas y procesados")
print(f"[GNN] feature-engineering (extraccion + combinaciones): {t_feateng:.2f} s")

# --- 4. Prediccion GNN por profundidad ---------------------------------------
if not SELECTION:
    raise SystemExit(
        "config.yaml -> gnn.selection esta vacio. Define depth -> {dataset_name, input_df} "
        "apuntando a los checkpoints GNN entrenados."
    )

# Coordenadas de referencia (cualquier df sirve; usamos el primero de la seleccion)
ref_key = next(iter(SELECTION.values()))["input_df"]
df_out = dfs[ref_key].loc[:, ["Date", "Latitude", "Longitude"]].copy()

n_pixels = len(df_out)
per_depth_time = {}
t_infer0 = time.perf_counter()
for depth, spec in SELECTION.items():
    dataset_name = spec["dataset_name"]
    input_df = spec["input_df"]
    df_in = dfs[input_df]
    print(f"[GNN] depth {depth} | dataset {dataset_name} | modelo {MODEL_NAME}")
    t0 = time.perf_counter()
    res = predict_map_gnn(
        df_pixels=df_in,
        model_dir=args.models,
        dataset_name=dataset_name,
        model_name=MODEL_NAME,
        knn_k=KNN_K,
        device=DEVICE,
    )
    dt = time.perf_counter() - t0
    per_depth_time[depth] = dt
    print(f"[GNN] inferencia depth {depth}: {dt:.2f} s")
    df_out[f"Chl_pred_{depth}"] = res["Chl_pred"].values
t_infer = time.perf_counter() - t_infer0

# --- Cronometraje: marcadores para el log + sidecar JSON para run_pipeline -----
print(f"INFER_TIME_S: {t_infer:.2f}")            # tiempo de inferencia GNN
print(f"FEATENG_TIME_S: {t_feateng:.2f}")        # tiempo de feature-engineering
print(f"INFER_PIXELS: {n_pixels}")
print(f"[GNN] inferencia total ({n_pixels} pixeles x {len(SELECTION)} profundidades): {t_infer:.2f} s")

# Sidecar con el desglose, que run_pipeline.py consolida en el CSV unico de tiempos.
try:
    sidecar = os.path.join(args.pred, f"{target_dates[0]}_timing.json")
    with open(sidecar, "w") as f:
        json.dump({
            "n_pixels": n_pixels,
            "feateng_s": round(t_feateng, 3),
            "infer_s": round(t_infer, 3),
            "infer_per_depth_s": {d: round(per_depth_time.get(d, float("nan")), 3)
                                  for d in ["0_1", "1_2", "2_3", "3_4"]},
            "model": MODEL_NAME, "knn_k": KNN_K, "device": DEVICE,
        }, f)
except OSError as e:
    print(f"[aviso] no se pudo escribir el sidecar de tiempos ({e})")

cols = ["Date", "Latitude", "Longitude", "Chl_pred_0_1", "Chl_pred_1_2", "Chl_pred_2_3", "Chl_pred_3_4"]
df_out.loc[:, cols].to_csv(f"{args.pred}/{target_dates[0]}_pred.csv", index=False)
print(f"Predicciones GNN guardadas en {args.pred}/{target_dates[0]}_pred.csv")
