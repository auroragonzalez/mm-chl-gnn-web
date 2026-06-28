"""cloud_filter.py — consulta de nubosidad (catálogo SentinelHub de Copernicus).

Devuelve, para una ventana de fechas alrededor de un objetivo, qué adquisiciones
Sentinel-2 L1C existen sobre la laguna y su % de nubes (`eo:cloud_cover`), y un
orden de fechas candidato que prioriza: (1) fechas con nubes <= umbral, las más
cercanas al objetivo primero; (2) como respaldo, el resto por cercanía.

Requiere credenciales OAuth de Copernicus (`CDS_ID`/`CDS_SECRET`). Si no hay
credenciales o el catálogo falla, las funciones lo señalan y el llamante debe
caer al método basado solo en S3.
"""

import datetime

# AOI de la laguna (W, S, E, N), igual que en check_dates.py.
AOI = [-0.86, 37.65, -0.74, 37.8]


def get_cloud_cover_map(start_date, end_date, cds_id, cds_secret):
    """Devuelve {date: cloud_cover_%} para las adquisiciones en [start, end].

    Lanza excepción si la consulta al catálogo falla (el llamante decide el
    fallback). Si una fecha tiene varias escenas se queda con la menos nubosa.
    """
    from sentinelhub import SentinelHubCatalog, DataCollection, SHConfig, CRS, BBox

    config = SHConfig()
    config.sh_client_id = cds_id
    config.sh_client_secret = cds_secret
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"

    catalog = SentinelHubCatalog(config=config)
    bbox = BBox(bbox=AOI, crs=CRS.WGS84)
    search = catalog.search(
        DataCollection.SENTINEL2_L1C,
        bbox=bbox,
        time=(start_date.isoformat(), end_date.isoformat()),
        fields={"include": ["id", "properties.eo:cloud_cover"], "exclude": []},
    )

    cloud: dict[datetime.date, float] = {}
    for item in search:
        # id tipo S2A_MSIL1C_20260620T...; la fecha son los 8 primeros del 3er campo.
        d = datetime.datetime.strptime(item["id"].split("_")[2][:8], "%Y%m%d").date()
        cc = float(item["properties"]["eo:cloud_cover"])
        if d not in cloud or cc < cloud[d]:
            cloud[d] = cc
    return cloud


def ranked_candidates(target, window, max_cloud, cloud_map):
    """Orden de fechas a intentar: aceptables (nubes<=umbral) por cercanía primero,
    luego el resto por cercanía como respaldo.

    `cloud_map` es {date: cloud_%}. Solo se consideran fechas dentro de +-window.
    """
    in_window = [
        d for d in cloud_map if abs((d - target).days) <= window
    ]
    acceptable = sorted(
        [d for d in in_window if cloud_map[d] <= max_cloud],
        key=lambda d: (abs((d - target).days), cloud_map[d]),
    )
    rest = sorted(
        [d for d in in_window if cloud_map[d] > max_cloud],
        key=lambda d: abs((d - target).days),
    )
    return acceptable + rest
