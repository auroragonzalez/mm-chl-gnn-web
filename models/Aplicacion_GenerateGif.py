import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import rasterio
import os
from matplotlib.colorbar import ColorbarBase
import argparse
import geopandas as gpd
import fiona

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, type=str, help="Fecha del producto a descargar (YYYY-MM-DD)")
parser.add_argument("--input", required=True, help="Directorio donde están los TIFFs generados a partir de CSVs")
parser.add_argument("--output", required=True, help="Directorio donde se guarda el gif generado")
parser.add_argument("--colormap", required=True, help="Fichero con el colormap a utilizar")
parser.add_argument("--bathymetry", required=True, help="Fichero con el mapa de batimetría del Mar Menor")
args = parser.parse_args()

date = args.date

# === Archivos tif a animar ===
tif_paths = [
    f'{args.input}{date}_chl_map_0_1.tif',
    f'{args.input}{date}_chl_map_1_2.tif',
    f'{args.input}{date}_chl_map_2_3.tif',
    f'{args.input}{date}_chl_map_3_4.tif'
]

# === Cargar batimetría (Python 3.11: habilitar driver KML explícitamente) ===
fiona.drvsupport.supported_drivers['KML'] = 'r'
bathymetry_gdf = gpd.read_file(args.bathymetry, driver="KML")
bathymetry_gdf = bathymetry_gdf.set_crs("EPSG:4326", allow_override=True)

# === Función para leer .tif como array ===
def read_tif_as_array(path):
    with rasterio.open(path) as src:
        return src.read(1), src.crs, [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# === Leer colormap personalizado desde .txt ===
def load_qgis_colormap(colormap_path):
    colors = []
    boundaries = []

    with open(colormap_path, "r") as f:
        for line in f:
            if line.startswith("#") or "INTERPOLATION" in line:
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            value = float(parts[0])
            r, g, b, a = [int(p) for p in parts[1:5]]
            boundaries.append(value)
            colors.append((r / 255, g / 255, b / 255, a / 255))

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=len(colors))
    return cmap, norm, boundaries


# === Cargar colormap personalizado ===
cmap, norm, boundaries = load_qgis_colormap(args.colormap)

# Ticks y etiquetas
tick_locs = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(boundaries)-1)]
labels = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0", "2.4", "2.8", "3.2", "3.6", "4.0", "4.5", "5.0", "6.0", "8.0", "10.0", "12.0", "15.0", "18.0", "24.0", "30.0"]
tick_labels = labels[:-1]

frames = []
for path in tif_paths:
    data, raster_crs, extent = read_tif_as_array(path)
    depth_str = path.replace(f"{args.input}{date}_chl_map_", "").replace(".tif", "").replace("_", "-")

    # Enmascarar NaN / nodata
    data = np.ma.masked_invalid(data)

    # Reproyectar batimetría al CRS del raster
    bathymetry_proj = bathymetry_gdf.to_crs(raster_crs)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin="upper")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # === Superponer líneas de batimetría ===
    bathymetry_proj.plot(ax=ax, color="white", linewidth=0.6, alpha=0.7)

    # === Etiquetas de batimetría ===
    for _, row in bathymetry_proj.iterrows():
        if row.geometry is None:
            continue
        pt = row.geometry.representative_point()
        ax.text(
            pt.x, pt.y,
            str(row.get("name", "")),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1),
        )

    ax.axis('off')

    # Texto con profundidad
    ax.text(-0.3, 0.9, f'Depth {depth_str}', color='white', fontsize=24, fontweight='bold',
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    # Colorbar personalizado (leyenda)
    cbar_ax = fig.add_axes([0.75, 0.15, 0.03, 0.75])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, boundaries=boundaries, ticks=tick_locs)
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=8)
    cb.ax.text(0.5, 1.02, "Chl mg/m³", fontsize=10, ha='center', va='bottom', transform=cb.ax.transAxes)

    # Convertir a imagen
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_rgb = img[..., :3]
    frames.append(Image.fromarray(img_rgb))
    plt.close(fig)

# Guardar el gif
frames[0].save(
    f'{args.output}{date}_chl_pred_loop.gif',
    save_all=True,
    append_images=frames[1:],
    duration=1000,
    loop=0
)