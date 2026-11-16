"""
Inference utilities for GNN-based Chl-a map generation.

Replaces ``predict_with_model()`` in ``Aplicacion_utils.py`` — the rest
of the map-generation pipeline (pixel extraction, band combinations,
TIFF generation, plotting) is reused as-is.

Typical usage
-------------
>>> from gnn.inference import predict_map_gnn
>>> df_result = predict_map_gnn(
...     df_pixels=df_pixels,        # from Aplicacion_utils.extract_pixels_in_marmenor
...     model_dir="results/gnn/depth_in_0_1/seed42",
...     dataset_name="C2RCC_rhow_15x15_depth_in_0_1",
...     model_name="GraphSAGE",
... )
>>> # df_result now contains a "Chl_pred" column
"""

import json
import logging
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data

from .models import GNN_MODELS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_gnn_model(
    model_dir: str,
    dataset_name: str,
    model_name: str,
    device: str = "cpu",
):
    """
    Load a trained GNN model from disk.

    Expects files created by ``train.run_gnn_pipeline``:
      - ``{dataset_name}_{model_name}.pt``       — state dict
      - ``{dataset_name}_{model_name}_config.json`` — architecture config
      - ``{dataset_name}_scaler.pkl``             — RobustScaler
      - ``{dataset_name}_feature_cols.json``      — feature column names

    Parameters
    ----------
    model_dir : str
        Directory containing the saved files (e.g.
        ``results/gnn/depth_in_0_1/seed42``).
    dataset_name : str
        Dataset key, e.g. ``"C2RCC_rhow_15x15_depth_in_0_1"``.
    model_name : str
        One of ``"GCN"``, ``"GAT"``, ``"GraphSAGE"``.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    tuple
        ``(model, scaler, feature_cols)``
    """
    # Load config
    config_path = os.path.join(
        model_dir, f"{dataset_name}_{model_name}_config.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    model_kwargs = config["model_kwargs"]
    model_cls = GNN_MODELS[config["model_name"]]
    model = model_cls(**model_kwargs)

    # Load weights
    weights_path = os.path.join(
        model_dir, f"{dataset_name}_{model_name}.pt"
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load scaler
    scaler_path = os.path.join(model_dir, f"{dataset_name}_scaler.pkl")
    scaler: Optional[RobustScaler] = None
    if os.path.isfile(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Load feature columns
    cols_path = os.path.join(model_dir, f"{dataset_name}_feature_cols.json")
    with open(cols_path, "r") as f:
        feature_cols = json.load(f)

    logger.info(
        f"Loaded {model_name} for {dataset_name} "
        f"({model.count_parameters():,} params)"
    )
    return model, scaler, feature_cols


# ──────────────────────────────────────────────────────────────────────
# Pixel-level prediction
# ──────────────────────────────────────────────────────────────────────

def _build_pixel_graph(
    coords: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """
    Build a k-NN spatial graph from pixel coordinates.

    Parameters
    ----------
    coords : np.ndarray of shape (N, 2)
        Latitude, Longitude (UTM) of each pixel.
    k : int
        Number of spatial neighbours.

    Returns
    -------
    np.ndarray of shape (2, E) — edge index.
    """
    k_actual = min(k, coords.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean")
    nn.fit(coords)
    _, neighbor_idx = nn.kneighbors(coords)

    edge_set: set[tuple[int, int]] = set()
    for i, neighbors in enumerate(neighbor_idx):
        for j in neighbors[1:]:
            edge_set.add((i, j))
            edge_set.add((j, i))

    return np.array(sorted(edge_set), dtype=np.int64).T


def _station_anchored_graph(
    pix_coords: np.ndarray,
    anchor_coords: np.ndarray,
    spatial_k: int = 5,
) -> np.ndarray:
    """
    Grafo de despliegue que reproduce los vecindarios del entrenamiento.

    En entrenamiento los vecinos espaciales de un nodo son las ``spatial_k``
    boyas mas cercanas del mismo dia (a 2-11 km). Aqui los pixeles de la
    imagen situados en las posiciones de las boyas ("anclas") se anaden como
    nodos extra; cada pixel recibe mensajes de sus ``spatial_k`` anclas mas
    cercanas, y las anclas se conectan entre si como las boyas en
    entrenamiento. Los pixeles no intercambian mensajes entre ellos: los
    pixeles adyacentes (10-60 m) tienen reflectancias casi identicas y el
    modelo no ve el contraste de vecindario con el que se entreno. En
    observaciones de boya no vistas, el grafo de pixeles adyacentes casi
    duplica el RMSLE; este grafo cuesta 0.03-0.07 (ver el articulo, sec. 7.2).

    Devuelve edge_index con las anclas indexadas despues de los P pixeles.
    """
    P, A = len(pix_coords), len(anchor_coords)
    k = min(spatial_k, A - 1)
    src, dst = [], []
    _, ia = NearestNeighbors(n_neighbors=k + 1).fit(anchor_coords).kneighbors(anchor_coords)
    for i in range(A):
        for j in ia[i, 1:]:
            src += [P + j, P + i]
            dst += [P + i, P + j]
    _, ip = NearestNeighbors(n_neighbors=k).fit(anchor_coords).kneighbors(pix_coords)
    src = np.concatenate([np.asarray(src, dtype=np.int64), (P + ip).ravel()])
    dst = np.concatenate([np.asarray(dst, dtype=np.int64), np.repeat(np.arange(P), k)])
    return np.stack([src, dst]).astype(np.int64)


def load_station_anchors(path: str) -> np.ndarray:
    """Posiciones de las boyas (UTM 30N, [northing, easting]) desde JSON."""
    with open(path) as f:
        d = json.load(f)
    return np.array([[s["northing"], s["easting"]] for s in d["stations"]],
                    dtype=np.float64)


@torch.no_grad()
def predict_pixels_gnn(
    model,
    features: np.ndarray,
    coords: np.ndarray,
    scaler: Optional[RobustScaler] = None,
    knn_k: int = 8,
    device: str = "cpu",
    batch_size: int = 50_000,
    graph_mode: str = "station",
    anchor_coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run GNN inference on pixel-level features.

    For very large images, inference is done in batches of
    ``batch_size`` pixels (each batch gets its own local k-NN graph).

    Parameters
    ----------
    model : nn.Module
        Trained GNN in eval mode.
    features : np.ndarray of shape (N, F)
        Raw features per pixel.
    coords : np.ndarray of shape (N, 2)
        Pixel coordinates (for graph construction).
    scaler : RobustScaler or None
        If provided, transform features before inference.
    knn_k : int
        k for the pixel spatial graph.
    device : str
        ``"cpu"`` or ``"cuda"``.
    batch_size : int
        Max pixels per inference batch.

    Returns
    -------
    np.ndarray of shape (N,) — predicted Chl-a values.
    """
    model.eval()
    model.to(device)

    # Scale features
    if scaler is not None:
        features = scaler.transform(features)

    features = features.astype(np.float32)
    N = features.shape[0]
    all_preds = np.zeros(N, dtype=np.float32)

    if graph_mode == "station":
        if anchor_coords is None:
            raise ValueError("graph_mode='station' necesita anchor_coords")
        # Features de las anclas = pixel de la imagen mas cercano a cada boya
        _, near = NearestNeighbors(n_neighbors=1).fit(coords).kneighbors(anchor_coords)
        anchor_feat = features[near[:, 0]]

    # Process in batches
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_feat = features[start:end]
        batch_coords = coords[start:end]

        if graph_mode == "station":
            edge_index = _station_anchored_graph(batch_coords, anchor_coords)
            x = np.concatenate([batch_feat, anchor_feat]).astype(np.float32)
        else:  # legacy: grafo k-NN de pixeles adyacentes
            edge_index = _build_pixel_graph(batch_coords, k=knn_k)
            x = batch_feat

        preds = model(torch.from_numpy(x).to(device),
                      torch.from_numpy(edge_index).to(device))
        all_preds[start:end] = preds[: end - start].cpu().numpy()

    # Clip to physical minimum (same as tabular project)
    all_preds = np.clip(all_preds, 0.2, None)
    return all_preds


def predict_pixels_gnn_with_uncertainty(
    model,
    features: np.ndarray,
    coords: np.ndarray,
    scaler: Optional[RobustScaler] = None,
    knn_k: int = 8,
    device: str = "cpu",
    batch_size: int = 50_000,
    n_mc_passes: int = 30,
) -> dict:
    """
    Like ``predict_pixels_gnn()`` but returns uncertainty via MC Dropout.

    Returns dict with keys: chl_mean, chl_std, chl_p05, chl_p95.
    """
    import torch.nn as nn

    model.eval()
    model.to(device)

    if scaler is not None:
        features = scaler.transform(features)

    features = features.astype(np.float32)
    N = features.shape[0]
    all_means = np.zeros(N, dtype=np.float32)
    all_stds = np.zeros(N, dtype=np.float32)
    all_p05 = np.zeros(N, dtype=np.float32)
    all_p95 = np.zeros(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_feat = features[start:end]
        batch_coords = coords[start:end]

        edge_index = _build_pixel_graph(batch_coords, k=knn_k)

        x = torch.from_numpy(batch_feat).to(device)
        ei = torch.from_numpy(edge_index).to(device)

        samples = []
        for _ in range(n_mc_passes):
            _enable_dropout(model)
            pred = model(x, ei)
            samples.append(pred.cpu().numpy())

        samples_arr = np.stack(samples, axis=0)
        all_means[start:end] = samples_arr.mean(axis=0)
        all_stds[start:end] = samples_arr.std(axis=0)
        all_p05[start:end] = np.percentile(samples_arr, 5, axis=0)
        all_p95[start:end] = np.percentile(samples_arr, 95, axis=0)

    return {
        "chl_mean": np.clip(all_means, 0.2, None),
        "chl_std": all_stds,
        "chl_p05": np.clip(all_p05, 0.2, None),
        "chl_p95": all_p95,
    }


def _enable_dropout(model):
    """Force all Dropout layers to train mode for MC inference."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


# ──────────────────────────────────────────────────────────────────────
# High-level map generation
# ──────────────────────────────────────────────────────────────────────

def predict_map_gnn(
    df_pixels: pd.DataFrame,
    model_dir: str,
    dataset_name: str,
    model_name: str,
    feature_cols: Optional[list[str]] = None,
    scaler: Optional[RobustScaler] = None,
    knn_k: int = 8,
    device: str = "cpu",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    uncertainty: bool = False,
    mc_passes: int = 30,
    graph_mode: str = "station",
    anchors_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict Chl-a for a DataFrame of pixels (from a Sentinel-2 image).

    This function is the GNN replacement for ``predict_with_model()``
    in ``Aplicacion_utils.py``.  The rest of the map pipeline
    (``extract_pixels_in_marmenor``, ``add_band_combinations``,
    ``Aplicacion_TIFFfromCSV.py``, ``Aplicacion_PlotTIFF.py``) stays
    the same.

    Parameters
    ----------
    df_pixels : DataFrame
        Pixel data with feature columns and coordinate columns.
    model_dir : str
        Path to saved model directory.
    dataset_name : str
        Dataset key.
    model_name : str
        GNN architecture name.
    feature_cols : list[str] or None
        Feature column names.  If None, loaded from disk.
    scaler : RobustScaler or None
        If None, loaded from disk.
    knn_k : int
        k for the pixel spatial graph.
    device : str
        ``"cpu"`` or ``"cuda"``.
    lat_col, lon_col : str
        Column names for coordinates.
    uncertainty : bool
        If True, also return Chl_pred_std, Chl_pred_p05, Chl_pred_p95
        using MC Dropout.
    mc_passes : int
        Number of MC forward passes when uncertainty=True.
    graph_mode : str
        ``"station"`` (por defecto): grafo anclado a las boyas.
        ``"pixel"``: grafo k-NN de pixeles adyacentes (version original).
    anchors_path : str or None
        JSON con las posiciones de las boyas; por defecto
        ``{model_dir}/station_anchors.json``.

    Returns
    -------
    DataFrame
        Copy of ``df_pixels`` with added ``"Chl_pred"`` column
        (and uncertainty columns if requested).
    """
    # Load model artefacts if not provided
    model, disk_scaler, disk_feature_cols = load_gnn_model(
        model_dir, dataset_name, model_name, device=device,
    )
    if feature_cols is None:
        feature_cols = disk_feature_cols
    if scaler is None:
        scaler = disk_scaler

    # Prepare inputs
    df_work = df_pixels.copy()
    season_cols = ["Invierno", "Primavera", "Verano", "Otoño"]
    for sc in season_cols:
        if sc not in df_work.columns:
            df_work[sc] = 0.0

    if "Date" in df_work.columns:
        import pandas as pd

        df_work["Date"] = pd.to_datetime(df_work["Date"])
        month = df_work["Date"].dt.month.iloc[0]
        for sc in season_cols:
            df_work[sc] = 0.0
        if month in (12, 1, 2):
            df_work["Invierno"] = 1.0
        elif month in (3, 4, 5):
            df_work["Primavera"] = 1.0
        elif month in (6, 7, 8):
            df_work["Verano"] = 1.0
        else:
            df_work["Otoño"] = 1.0

    available = [c for c in feature_cols if c in df_work.columns]
    missing = [c for c in feature_cols if c not in df_work.columns]
    if missing:
        logger.warning(
            f"Missing {len(missing)} feature columns, filling with 0: "
            f"{missing[:5]}..."
        )
        for c in missing:
            df_work[c] = 0.0

    features = df_work[feature_cols].values.astype(np.float32)
    coords = df_work[[lat_col, lon_col]].values.astype(np.float64)

    # Sanitiza valores no finitos antes de escalar/propagar. Las combinaciones de
    # bandas (cocientes, 1/banda) generan NaN/Inf donde rhow es 0 o negativo
    # (típico en píxeles turbios/de borde/nube). Como la GNN propaga por mensajes,
    # un solo NaN contaminaría todo el grafo y dejaría el mapa entero en NaN.
    n_bad = int((~np.isfinite(features)).sum())
    if n_bad:
        logger.warning(
            f"{n_bad} valores no finitos (NaN/Inf) en las features de "
            f"{int((~np.isfinite(features)).any(axis=1).sum())} píxeles; "
            f"se sustituyen por 0 para evitar la propagación de NaN."
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    anchor_coords = None
    if graph_mode == "station":
        anchor_coords = load_station_anchors(
            anchors_path or os.path.join(model_dir, "station_anchors.json"))
        if uncertainty:
            raise NotImplementedError(
                "MC Dropout con el grafo anclado no esta implementado; "
                "usa graph_mode='pixel' o intervalos conformales.")

    # Run inference
    if uncertainty:
        preds = predict_pixels_gnn_with_uncertainty(
            model, features, coords,
            scaler=scaler, knn_k=knn_k, device=device,
            n_mc_passes=mc_passes,
        )
        result = df_pixels.copy()
        result["Chl_pred"] = preds["chl_mean"]
        result["Chl_pred_std"] = preds["chl_std"]
        result["Chl_pred_p05"] = preds["chl_p05"]
        result["Chl_pred_p95"] = preds["chl_p95"]
    else:
        preds = predict_pixels_gnn(
            model, features, coords,
            scaler=scaler, knn_k=knn_k, device=device,
            graph_mode=graph_mode, anchor_coords=anchor_coords,
        )
        result = df_pixels.copy()
        result["Chl_pred"] = preds

    return result
