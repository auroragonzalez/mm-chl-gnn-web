import boto3
from botocore.config import Config
import datetime
import os
import csv
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, type=str, help="Fecha del producto a descargar (YYYY-MM-DD)")
parser.add_argument("--output", required=False, help="Directorio donde guardar el .SAFE")
parser.add_argument("--window", required=False, type=int, default=0,
                    help="Si la fecha exacta no tiene producto, buscar la adquisicion mas "
                         "cercana en +-N dias (0 = solo la fecha exacta).")
parser.add_argument("--max-cloud", required=False, type=float, default=100.0,
                    help="Umbral de nubosidad (%%). Con credenciales CDS, se priorizan las "
                         "fechas con nubes <= umbral (100 = no filtrar por nubes).")
args = parser.parse_args()

# Credenciales y configuración de sesión
ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_SECRET_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    raise EnvironmentError("No se encontraron los credenciales de S3 en el entorno o .env")

# Timeouts amplios + reintentos: el endpoint de Copernicus a veces va lento y el
# handshake/lectura puede tardar; reintentar evita fallos transitorios.
_boto_cfg = Config(
    connect_timeout=30,
    read_timeout=120,
    retries={"max_attempts": 5, "mode": "standard"},
)

session = boto3.session.Session()
s3 = session.resource(
    's3',
    endpoint_url='https://eodata.dataspace.copernicus.eu',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='default',
    config=_boto_cfg,
)

bucket = s3.Bucket("eodata")

target_date = datetime.datetime.strptime(str(args.date), "%Y-%m-%d").date()
WINDOW = max(0, int(args.window))
MAX_CLOUD = float(args.max_cloud)
CDS_ID = os.getenv("CDS_ID")
CDS_SECRET = os.getenv("CDS_SECRET")

# Tile ID a buscar
tile_id = "30SXG"

# Prefijos base de búsqueda: L1C_N0500 o L1C
#prefixes = [f"Sentinel-2/MSI/L1C_N0500/{fecha.year}/{fecha.month:02d}/{fecha.day:02d}/" for fecha in fechas]
#prefixes = [f"Sentinel-2/MSI/L1C/{fecha.year}/{fecha.month:02d}/{fecha.day:02d}/" for fecha in fechas]

# Carpeta local destino
output_dir = args.output

prefix_patterns = [
    "Sentinel-2/MSI/L1C_N0500/{y}/{m:02d}/{d:02d}/",
    "Sentinel-2/MSI/L1C/{y}/{m:02d}/{d:02d}/"
]

def download_product(bucket, prefix: str, tile_id: str, target_root: str):
    """
    Busca y descarga un producto SAFE que contenga el tile_id en el prefijo dado.
    Devuelve el nombre del producto descargado o None si no encuentra nada.
    """
    found = False
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith('.SAFE/') and tile_id in obj.key:
            found = True
            product_prefix = obj.key
            print(f"Producto encontrado: {product_prefix}")
            local_target = os.path.join(target_root, os.path.basename(product_prefix.rstrip('/')))
            for file_obj in bucket.objects.filter(Prefix=product_prefix):
                local_path = os.path.join(local_target, os.path.relpath(file_obj.key, product_prefix))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if not file_obj.key.endswith('/'):  # evitar carpetas virtuales
                    print(f"Descargando {file_obj.key} → {local_path}")
                    bucket.download_file(file_obj.key, local_path)
            return product_prefix
    if not found:
        print(f"No se encontró producto SAFE con tile {tile_id} en {prefix}")
    return None



def candidate_dates(target, window):
    """Genera fechas por distancia creciente al objetivo: target, -1, +1, -2, +2, ...

    Para empates de distancia se prueba primero la fecha anterior."""
    yield target
    for delta in range(1, window + 1):
        yield target - datetime.timedelta(days=delta)
        yield target + datetime.timedelta(days=delta)


def build_candidate_order():
    """Decide el orden de fechas a intentar.

    Con credenciales CDS y umbral de nubes < 100, consulta el catálogo y prioriza
    las fechas poco nubosas más cercanas (con respaldo a las nubosas por cercanía).
    Si no hay credenciales o el catálogo falla, usa el orden por cercanía simple.
    Devuelve (lista_de_fechas, cloud_map | None).
    """
    use_cloud = WINDOW >= 0 and MAX_CLOUD < 100.0 and CDS_ID and CDS_SECRET
    if not use_cloud:
        if MAX_CLOUD < 100.0 and not (CDS_ID and CDS_SECRET):
            print("[nubes] Sin credenciales CDS: no se puede filtrar por nubosidad; "
                  "se usa solo cercanía.")
        return list(candidate_dates(target_date, WINDOW)), None

    try:
        from cloud_filter import get_cloud_cover_map, ranked_candidates
        start = target_date - datetime.timedelta(days=WINDOW)
        end = target_date + datetime.timedelta(days=WINDOW)
        print(f"[nubes] Consultando nubosidad en [{start} .. {end}] (umbral {MAX_CLOUD:.0f}%)")
        cloud_map = get_cloud_cover_map(start, end, CDS_ID, CDS_SECRET)
        if not cloud_map:
            print("[nubes] El catálogo no devolvió adquisiciones; se usa solo cercanía.")
            return list(candidate_dates(target_date, WINDOW)), None
        for d in sorted(cloud_map):
            print(f"[nubes]   {d}: {cloud_map[d]:.0f}% nubes")
        order = ranked_candidates(target_date, WINDOW, MAX_CLOUD, cloud_map)
        if not order:
            print("[nubes] Ninguna adquisición en la ventana; se usa solo cercanía.")
            return list(candidate_dates(target_date, WINDOW)), None
        return order, cloud_map
    except Exception as exc:  # noqa: BLE001
        print(f"[nubes] Fallo la consulta de nubosidad ({exc}); se usa solo cercanía.")
        return list(candidate_dates(target_date, WINDOW)), None


def main():
    os.makedirs(output_dir, exist_ok=True)

    if WINDOW > 0:
        print(f"Buscando la adquisicion mas cercana a {target_date} en +-{WINDOW} dias")

    order, cloud_map = build_candidate_order()

    resolved = None
    for fecha in order:
        cc = f" (nubes {cloud_map[fecha]:.0f}%)" if cloud_map and fecha in cloud_map else ""
        print(f"\nProcesando fecha {fecha.strftime('%Y-%m-%d')}{cc}")
        for pattern in prefix_patterns:
            prefix = pattern.format(y=fecha.year, m=fecha.month, d=fecha.day)
            print(f"Buscando productos en: {prefix}")
            result = download_product(bucket, prefix, tile_id, output_dir)
            if result:
                resolved = fecha
                break
        if resolved:
            break

    if resolved:
        resolved_str = resolved.strftime("%Y-%m-%d")
        cloud = cloud_map.get(resolved) if cloud_map else None
        # Persistir la fecha realmente usada para que el resto del pipeline la lea.
        with open(os.path.join(output_dir, "resolved_date.txt"), "w") as f:
            f.write(resolved_str)
        # Marcadores parseables (los capturan run_pipeline.py y webapp.py).
        print(f"RESOLVED_DATE: {resolved_str}")
        if cloud is not None:
            print(f"RESOLVED_CLOUD: {cloud:.0f}")
        cloud_txt = f" (nubes {cloud:.0f}%)" if cloud is not None else ""
        if resolved != target_date:
            print(f"La fecha solicitada {target_date} no tenia producto idóneo; "
                  f"se usa la adquisicion mas cercana: {resolved_str}{cloud_txt}")
        else:
            print(f"Se descargó el producto para la fecha solicitada{cloud_txt}.")
        if cloud is not None and cloud > MAX_CLOUD:
            print(f"AVISO: ninguna adquisición en +-{WINDOW} días baja del umbral "
                  f"de nubes ({MAX_CLOUD:.0f}%); la elegida tiene {cloud:.0f}%.")
        sys.exit(0)
    else:
        if WINDOW > 0:
            print(f"\nNo se encontró ningún producto en +-{WINDOW} días alrededor de {target_date}.")
        else:
            print("\nNo se descargó ningún producto.")
        sys.exit(1)


if __name__ == "__main__":
    main()
