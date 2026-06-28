import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, type=str, help="Fecha del producto a descargar (YYYY-MM-DD)")
parser.add_argument("--input", required=False, help="Directorio donde están guardados los .SAFE")
args = parser.parse_args()

# Ruta donde están las carpetas SAFE descargadas
input_dir = args.input

# Recorremos todos los elementos en la carpeta
for item in os.listdir(input_dir):
    item_path = os.path.join(input_dir, item)

    # Solo procesar carpetas que acaban en .SAFE
    if os.path.isdir(item_path) and item.endswith(".SAFE"):
        zip_path = os.path.join(input_dir, item + ".zip")
        
        if not os.path.exists(zip_path):
            print(f"Comprimiendo {item_path} → {zip_path}")
            shutil.make_archive(base_name=zip_path[:-4], format="zip", root_dir=input_dir, base_dir=item)
        else:
            print(f"{zip_path} ya existe. Saltando.")