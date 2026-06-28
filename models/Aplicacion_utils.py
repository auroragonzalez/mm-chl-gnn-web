
import pandas as pd
import os
from itertools import permutations, combinations
import re
import numpy as np
import rasterio
from rasterio import features
import numpy as np
import geopandas as gpd
import json
from typing import Optional
from joblib import load as joblib_load


def extract_pixels_in_marmenor(folder_path, target_dates, grouping, net_set, polygon_path):
    """
    Extrae valores de píxeles dentro del área del Mar Menor a partir de GeoTIFFs y un polígono de máscara.

    Args:
        folder_path: carpeta con los TIFFs
        target_dates: lista de fechas (formato YYYY-MM-DD)
        grouping: tamaño de agrupamiento ("1x1", "3x3", "5x5", etc.)
        net_set: prefijo del conjunto (C2X-Complex, C2X, C2RCC)
        polygon_path: ruta al archivo GeoJSON o Shapefile del Mar Menor

    Returns:
        pd.DataFrame con bandas y coordenadas por píxel
    """
    results = []
    target_dates = sorted(set(target_dates))

    # Selección de ficheros TIFF
    if net_set == "C2X-Complex":
        tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and 'C2XComplexNets' in f]
    elif net_set == "C2X":
        tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and 'C2XNets' in f]
    elif net_set == "C2RCC":
        tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif') and 'C2RCC' in f]
    else:
        raise ValueError("Net Set no reconocido")

    # Leer el polígono del Mar Menor
    #gdf = gpd.read_file(polygon_path)
    with open(polygon_path, "r") as f:
        data = json.load(f)

    gdf = gpd.GeoDataFrame.from_features(data["features"])
    if gdf.crs is None:
        gdf.set_crs("EPSG:32630", inplace=True)


    for date in target_dates:
        date_str = date.replace('-', '')
        matching = [f for f in tif_files if date_str in f]
        if not matching:
            continue

        tiff_file = os.path.join(folder_path, matching[0])
        with rasterio.open(tiff_file) as dataset:
            print(f"Procesando {tiff_file}")
            bands = dataset.read()  # shape: (n_bands, height, width)

            # Crear máscara booleana de los píxeles dentro del polígono
            mask = features.geometry_mask(
                geometries=gdf.geometry,
                transform=dataset.transform,
                invert=True,
                out_shape=(dataset.height, dataset.width)
            )

            if grouping == "3x3":
                offset = 1
            elif grouping == "5x5":
                offset = 2
            elif grouping == "9x9":
                offset = 4
            elif grouping == "15x15":
                offset = 7
            else:
                offset = 0

            # Iterar sobre todos los píxeles dentro del polígono
            idxs = np.argwhere(mask)

            for row_idx, col_idx in idxs:
                try:
                    if offset > 0:
                        row_start = max(row_idx - offset, 0)
                        row_end = min(row_idx + offset + 1, dataset.height)
                        col_start = max(col_idx - offset, 0)
                        col_end = min(col_idx + offset + 1, dataset.width)

                        window = (
                            slice(row_start, row_end),
                            slice(col_start, col_end)
                        )

                        reflectances = bands[:, window[0], window[1]]
                        values = np.median(reflectances, axis=(1, 2))
                    else:
                        values = bands[:, row_idx, col_idx]

                    # Convertir coordenadas a UTM (o lat/lon si prefieres)
                    lon, lat = dataset.xy(row_idx, col_idx)

                    results.append({
                        "Date": date,
                        "Latitude": lat,
                        "Longitude": lon,
                        **{f"Band_{i+1}": val for i, val in enumerate(values)}
                    })
                except Exception as e:
                    print(f"Error en píxel ({row_idx},{col_idx}): {e}")
                    continue

    return pd.DataFrame(results)


def create_processed_dfs(dfs_tifs):

    band_names = {
        "Band_1": "rtoa_B1",
        "Band_2": "rtoa_B2",
        "Band_3": "rtoa_B3",
        "Band_4": "rtoa_B4",
        "Band_5": "rtoa_B5",
        "Band_6": "rtoa_B6",
        "Band_7": "rtoa_B7",
        "Band_8": "rtoa_B8",
        "Band_9": "rtoa_B8A",
        "Band_10": "rtoa_B9",
        "Band_11": "rtoa_B10",
        "Band_12": "rtoa_B11",
        "Band_13": "rtoa_B12",
        "Band_14": "rhow_B1",
        "Band_15": "rhow_B2",
        "Band_16": "rhow_B3",
        "Band_17": "rhow_B4",
        "Band_18": "rhow_B5",
        "Band_19": "rhow_B6",
        "Band_20": "rhow_B7",
        "Band_21": "rhow_B8A"
    }

    dfs_tifs_all = {} 

    for nombre_df, df in list(dfs_tifs.items()):
        # Rename de todas la columnas
        df = df.rename(columns=band_names)

        # Para coger de solamente de C2RCC las bandas TOA, puesto que son iguales en C2X y Complex
        if "15x15" in nombre_df:
        #     #print(nombre_df)
            ventana = nombre_df.split("_")[3]
            dfs_tifs_all[f"df_tifs_TOA_{ventana}"] = df.iloc[:,np.r_[0:16]]


        name_chunks = nombre_df.split("_")
        # El resto de columnas igual
        dfs_tifs_all[f"{name_chunks[0]}_{name_chunks[1]}_{name_chunks[2]}_rhow_{name_chunks[3]}"] = df.iloc[:,np.r_[0:3, 16:24]]
    
    return dfs_tifs_all


def diferencia_normalizada(band1, band2):
    value = (band1 - band2)/(band1 + band2)
    return value.round(3)

def dall_gitelson(band1, band2, band3):
    value = (1/(band1) - 1/(band2))*(band3)
    return value.round(3)

def diferencia_normalizada_4bandas(band1, band2, band3, band4):
    value = (band1 - band2)/(band3 + band4)
    return value.round(3)

def diferencia_inversas(band1, band2):
    value = 1/(band1) - 1/(band2)
    return value.round(3)

def diferencia_relacion_4bandas(band1, band2, band3, band4):
    value = band1/band2 - band3/band4
    return value.round(3)

def suma_normalizada_3bandas(band1, band2, band3):
    value = (band1 + band3)/(band1 + band2)
    return value.round(3)

def add_two_band_difs(data, bands):
    for i, band1 in enumerate(bands):
        for band2 in bands[i+1:]:
            colname_dif_norm = f"dif_norm_{band1}_{band2}"
            data[colname_dif_norm] = diferencia_normalizada(data[band1], data[band2])
            #index_list.append(colname_dif_norm)
            colname_dif_inv = f"dif_inv_{band1}_{band2}"
            data[colname_dif_inv] = diferencia_inversas(data[band1], data[band2])
            #index_list.append(colname_dif_inv)
    return data

def add_dall_gitelson(data, bands):
    for band1, band2 in combinations(bands, 2):  # evita repeticiones de pares
        for band3 in bands:
            if band3 not in (band1, band2):  # evitar que band3 sea igual a los anteriores
                colname = f"dall_gitelson_{band1}_{band2}_{band3}"
                data[colname] = dall_gitelson(data[band1], data[band2], data[band3])
                #index_dall_gitelson_list.append(colname)
    return data

def add_norm_dif_4bands(data, bands):
    for band1, band2 in combinations(bands, 2):  # evita invertir band1 y band2
        for band3, band4 in combinations(bands, 2):  # evita invertir band3 y band4
            # Asegurar que todas las bandas son distintas
            if len({band1, band2, band3, band4}) == 4:
                colname = f"dif_norm_4_bands_{band1}_{band2}_{band3}_{band4}"
                data[colname] = diferencia_normalizada_4bandas(
                    data[band1], data[band2], data[band3], data[band4]
                )
                #index_dif_norm_4bands_list.append(colname)
    return data

def add_index_dif_rel_4bands(data, bands):
    for band1, band2, band3, band4 in permutations(bands, 4):
        # Evitar redundancias por simetría de términos
        # Criterio: solo aceptamos combinaciones donde el primer término es "menor" que el segundo
        if (band1, band2) < (band3, band4):  # evita generar la versión espejo con signo opuesto
            colname = f"dif_rel_4bands_{band1}_{band2}_{band3}_{band4}"
            data[colname] = diferencia_relacion_4bandas(
                data[band1], data[band2], data[band3], data[band4]
            )
            #index_dif_rel_4bands_list.append(colname)
    return data

def add_index_sum_norm_3bands(data, bands):
    band1, band2, band3, band4 = bands

    colname = f"sum_norm_3bands_{band1}_{band3}_{band2}"
    data[colname] = suma_normalizada_3bandas(data[band1], data[band2], data[band3])

    colname = f"sum_norm_3bands_{band2}_{band4}_{band3}"
    data[colname] = suma_normalizada_3bandas(data[band2], data[band3], data[band4])
       
    return data


def add_band_combinations(dfs_tifs_all):
    band_sets = [
        ['rtoa_B2', 'rtoa_B3', 'rtoa_B4', 'rtoa_B5'],
        ['rhow_B2', 'rhow_B3', 'rhow_B4', 'rhow_B5']
    ]

    dfs = dfs_tifs_all.copy()
    for nombre_df, df in dfs.items():
        for bands_to_use in band_sets:
            if set(bands_to_use).issubset(df.columns):
                df = add_two_band_difs(df, bands_to_use)
                df = add_dall_gitelson(df, bands_to_use)
                df = add_norm_dif_4bands(df, bands_to_use)
                df = add_index_dif_rel_4bands(df, bands_to_use)
                df = add_index_sum_norm_3bands(df, bands_to_use)
                
        dfs[nombre_df] = df 
    
    return dfs


def compactar_prefijos_columnas(df):
    nuevo_nombre_columnas = {}

    for col in df.columns:
        # Detectar columnas con patrones tipo index_algo_rhow_B1_rhow_B2_...
        if re.search(r'(rhow|rhown|rtoa)(_B\d+)+', col):
            partes = col.split('_')
            base = []
            bandas = []
            prefijo = None

            for parte in partes:
                if parte in ['rhow','rtoa']:
                    if not prefijo:
                        prefijo = parte
                elif parte.startswith('B'):
                    bandas.append(parte)
                else:
                    base.append(parte)

            if prefijo and bandas:
                #nuevo_nombre = f"{'_'.join(base)}_{prefijo}_{'_'.join(bandas)}"
                if base:
                    nuevo_nombre = f"{'_'.join(base)}_{prefijo}_{'_'.join(bandas)}"
                else:
                    nuevo_nombre = f"{prefijo}_{'_'.join(bandas)}"

                nuevo_nombre_columnas[col] = nuevo_nombre

    # Renombrar columnas
    df = df.rename(columns=nuevo_nombre_columnas)
    return df

def add_season(dfs):

    seasons = ['Invierno', 'Primavera', 'Verano', 'Otoño']

    for nombre_df, df in dfs.items():
        df = df.copy()

        # Estación desde la fecha
        df['Date'] = pd.to_datetime(df['Date'])
        m = df['Date'].dt.month
        df['Season'] = np.select(
            [m.isin([12,1,2]), m.isin([3,4,5]), m.isin([6,7,8])],
            ['Invierno', 'Primavera', 'Verano'],
            default='Otoño'
        )

        # One-hot con el conjunto fijo de categorías
        df['Season'] = pd.Categorical(df['Season'], categories=seasons)
        season_dummies = pd.get_dummies(df['Season']).reindex(columns=seasons, fill_value=0)
        season_dummies = season_dummies.astype('int8')  # opcional

        # (si quieres seguir convirtiendo otros object a category, hazlo aquí)
        for col in df.select_dtypes(include='object').columns:
            if col != 'Season':
                df[col] = df[col].astype('category')

        # Sustituye Season por las 4 columnas
        df = pd.concat([df.drop(columns=['Season']), season_dummies], axis=1)
        dfs[nombre_df] = df

        # Para rellenar algunos nulos que aparecen y que todos los df tengan las mismas filas
        dfs[nombre_df] = df.ffill()

    return dfs


# Cargadores específicos 
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


def _stem_and_keys(fname: str):
    """
    Obtiene el 'stem' base (sin sufijo) y separa dataset y modelo.
    E.g. 'TOA_9x9_depth_in_2_3_KNN_model.joblib' ->
         stem='TOA_9x9_depth_in_2_3_KNN', dataset='TOA_9x9_depth_in_2_3', model='KNN'
    """
    stem = (fname
            .replace("_model.joblib", "")
            .replace("_model.json", "")
            .replace("_model.cbm", "")
            .replace("_features.json", "")
            .replace("_metadata.json", ""))
    # dataset = todo menos el último bloque; model = último bloque
    parts = stem.split("_")
    model = parts[-1]
    dataset = "_".join(parts[:-1]) if len(parts) > 1 else stem
    return stem, dataset, model


def discover_artifacts(models_dir: str):
    """
    Explora la carpeta y devuelve un dict:
    artifacts[dataset][model] = {'model_path': ..., 'features_path': ..., 'metadata_path': ..., 'format': ...}
    """
    artifacts = {}
    for fname in os.listdir(models_dir):
        if not any(fname.endswith(suf) for suf in ["_model.joblib", "_model.json", "_model.cbm",
                                                   "_features.json", "_metadata.json"]):
            continue

        stem, dataset, model = _stem_and_keys(fname)
        artifacts.setdefault(dataset, {}).setdefault(model, {})
        entry = artifacts[dataset][model]

        fpath = os.path.join(models_dir, fname)
        if fname.endswith("_model.joblib"):
            entry["model_path"] = fpath
            entry["format"] = "joblib"
        elif fname.endswith("_model.json"):
            entry["model_path_json"] = fpath
            entry["format"] = entry.get("format", "json")
        elif fname.endswith("_model.cbm"):
            entry["model_path_cbm"] = fpath
            entry["format"] = "cbm"
        elif fname.endswith("_features.json"):
            entry["features_path"] = fpath
        elif fname.endswith("_metadata.json"):
            entry["metadata_path"] = fpath

    return artifacts


def load_features_list(features_path: str):
    """
    Lee *_features.json. Se espera una lista de nombres de columnas o un dict con la clave 'features'.
    """
    with open(features_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # admite tanto {'features': [...]} como {'feature_names': [...]}
        for k in ("features", "feature_names", "columns"):
            if k in data and isinstance(data[k], list):
                return data[k]
        # si viniese otra estructura, último recurso: llaves del dict si parecen nombres
        return list(data.keys())
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Formato de features no soportado en {features_path}")


def load_model_for_entry(dataset: str, model_name: str, entry: dict):
    """
    Carga el modelo según el formato disponible en 'entry'.
    - Preferencia: joblib (suele incluir Pipeline)
    - XGBoost JSON: crea un XGBRegressor y load_model(json_path)
    - CatBoost CBM: CatBoostRegressor().load_model(cbm_path)
    """
    # 1) Joblib (scikit-learn / pipeline)
    if "model_path" in entry and entry.get("format") == "joblib":
        return joblib_load(entry["model_path"])

    # 2) XGBoost JSON
    if entry.get("format") == "json" and "model_path_json" in entry:
        if XGBRegressor is None:
            raise ImportError("xgboost no está disponible en el entorno.")
        model = XGBRegressor()
        model.load_model(entry["model_path_json"])
        return model

    # 3) CatBoost CBM
    if entry.get("format") == "cbm" and "model_path_cbm" in entry:
        if CatBoostRegressor is None:
            raise ImportError("catboost no está disponible en el entorno.")
        model = CatBoostRegressor(verbose=False)
        model.load_model(entry["model_path_cbm"])
        return model

    raise FileNotFoundError(f"No se pudo cargar modelo para {dataset}/{model_name}: {entry}")



def predict_with_model(
    df: pd.DataFrame,
    models_dir: str,
    dataset_name: str,
    model_name: str,
    clip_min: Optional[float] = None,
    strict: bool = False
    ) -> pd.Series:
    """
    Aplica el modelo indicado a un DataFrame, usando los artefactos guardados en models_dir.

    Args:
        df: DataFrame de entrada con las columnas necesarias.
        models_dir: Carpeta donde están los archivos *_model.*, *_features.json, etc.
        dataset_name: Nombre del dataset usado en los archivos (sin extensión).
        model_name: Nombre del modelo ('XGB', 'CAT', etc.).
        clip_min: Valor mínimo a aplicar con np.clip() (p. ej. 0.3).
        strict: Si True, lanza error si faltan columnas. Si False, las ignora con aviso.

    Returns:
        pd.Series con las predicciones (misma longitud que df).
    """
    artifacts = discover_artifacts(models_dir)

    if dataset_name not in artifacts or model_name not in artifacts[dataset_name]:
        raise ValueError(f"No se encontró {dataset_name}/{model_name} en {models_dir}")

    entry = artifacts[dataset_name][model_name]

    print(f"Inference with {model_name} for {dataset_name}")

    # cargar modelo
    model = load_model_for_entry(dataset_name, model_name, entry)

    # cargar features
    if "features_path" in entry:
        features = load_features_list(entry["features_path"])
    elif hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise ValueError(f"No se pueden determinar las columnas requeridas para {dataset_name}/{model_name}")

    # verificar columnas
    missing = [c for c in features if c not in df.columns]
    if missing:
        msg = f"Faltan columnas en df: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        if strict:
            raise KeyError(msg)
        print(f"[AVISO] {msg}")

    # seleccionar features
    X = df[features].copy()

    # convertir objetos a numérico si aplica
    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col])
            except Exception:
                pass

    # predecir
    try:
        y_hat = model.predict(X)
    except TypeError:
        y_hat = model.predict(X.values)

    y_hat = np.asarray(y_hat).ravel()
    if clip_min is not None:
        y_hat = np.clip(y_hat, clip_min, None)

    return pd.Series(y_hat, index=df.index, name=f"{dataset_name}__{model_name}")
