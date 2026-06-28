# Mar Menor — Web de mapas de clorofila-a (GNN)

Servicio web autocontenido que genera mapas de clorofila-a (Chl-a) de la laguna
del Mar Menor a partir de imágenes Sentinel-2, usando los modelos de **redes
neuronales de grafos (GNN)** del proyecto principal.

El usuario solo elige una **fecha**; el servidor:

1. descarga el producto Sentinel-2 de Copernicus,
2. aplica la corrección atmosférica **C2RCC** con SNAP,
3. extrae los píxeles de la laguna y calcula las features,
4. **predice la Chl-a con la GNN** (`gnn/inference.predict_map_gnn`),
5. genera los GeoTIFF y los mapas, y los muestra en el navegador.

> Este repo es independiente del repo de investigación. El código de la GNN
> (`gnn/models.py`, `gnn/inference.py`) está **vendorizado** (copiado) aquí para
> que el despliegue no dependa del repo privado.

## Estructura

```
.
├── webapp.py                 # interfaz web (FastAPI)
├── run_pipeline.py           # orquestador: descarga → SNAP → GNN → TIFF → plots → GIF
├── config.yaml               # rutas + configuración de la GNN (selección por profundidad)
├── gnn/                      # ★ copia vendorizada del código GNN (solo inferencia)
│   ├── models.py
│   └── inference.py
├── models/                   # pasos del pipeline
│   ├── Aplicacion_Modelos_GNN.py   # predicción con la GNN (sustituye al paso tabular)
│   ├── Aplicacion_utils.py         # extracción de píxeles + features
│   ├── Aplicacion_TIFFfromCSV.py
│   ├── Aplicacion_PlotTIFF.py
│   └── Aplicacion_GenerateGif.py
├── fetch/                    # descarga Sentinel-2 + grafo SNAP C2RCC + geojson/kml/colormap
├── gnn_models/               # (vacío) checkpoints GNN — se montan en runtime
├── check_dates.py            # comprobar nubosidad por intervalo de fechas
├── Dockerfile                # Ubuntu + Python 3.11 + SNAP 12 + deps
└── requirements.txt
```

## Requisitos previos

1. **Credenciales de Copernicus (S3).** Por privacidad, **no se guardan en el
   servidor ni en un `.env`**: el usuario las introduce en el propio formulario web
   al generar un mapa (`S3_ACCESS_KEY` y `S3_SECRET_KEY`). Se inyectan únicamente en
   el entorno del subproceso de esa consulta y se descartan al terminar; no se
   escriben a disco, ni se registran en el log, ni se guardan en el estado del job.
   Opcionalmente, la casilla *"Recordar en este navegador"* las almacena en el
   `localStorage` del navegador del usuario (cliente), nunca en el servidor.

   > Para el modo CLI (sin web) sí hay que exportarlas como variables de entorno
   > (`-e S3_ACCESS_KEY=... -e S3_SECRET_KEY=...`), ver más abajo.
2. **Checkpoints GNN entrenados**. Para cada dataset de `config.yaml → gnn.selection`
   se esperan, dentro de la carpeta montada en `gnn.model_dir` (`/app/gnn_models`):
   ```
   {dataset_name}_{model_name}.pt
   {dataset_name}_{model_name}_config.json
   {dataset_name}_scaler.pkl
   {dataset_name}_feature_cols.json
   ```
   Estos los produce el entrenamiento del repo principal (`results/gnn/...`). Ajusta
   `gnn.selection` para que cada profundidad apunte al `dataset_name` correcto y al
   `input_df` de features correspondiente.

## Uso

```bash
# 1. Construir la imagen (la primera vez tarda: descarga SNAP + deps)
docker build -t chlwebapp .

# 2. Lanzar la web (las credenciales se introducen en el navegador, no aquí)
docker run -it --rm \
  -p 8000:8000 \
  -v "$(pwd)/gnn_models":/app/gnn_models \
  -v "$(pwd)/data/Chl_Maps":/app/data/Chl_Maps \
  chlwebapp
# abrir http://localhost:8000
```

Cada consulta crea un *job* en segundo plano: la página se autorrefresca y muestra
el log del pipeline en vivo hasta que aparecen los mapas (descarga + SNAP tardan
varios minutos). Las claves S3 se piden en el formulario y se usan solo para esa
consulta. Conviene comprobar antes que la fecha no tiene nubes (esta utilidad sí
usa variables de entorno, incluidas las OAuth `CDS_*`):

```bash
docker run -it --rm \
  -e CDS_ID=... -e CDS_SECRET=... chlwebapp \
  python3 check_dates.py --startdate 2025-10-01 --enddate 2025-10-16
```

### CLI (sin web)

El modo CLI no tiene formulario, así que las claves S3 se pasan como variables de
entorno (solo viven en ese contenedor efímero):

```bash
docker run -it --rm \
  -e S3_ACCESS_KEY=... -e S3_SECRET_KEY=... \
  -v "$(pwd)/gnn_models":/app/gnn_models \
  -v "$(pwd)/data/Chl_Maps":/app/data/Chl_Maps \
  chlwebapp python3 run_pipeline.py --date 2022-07-14
```

## Notas

- La predicción GNN corre en CPU por defecto (`config.yaml → gnn.device`).
- Predicciones recortadas a ≥ 0.2 mg/m³ (la cabeza de la GNN termina en Softplus).
- `k=8` vecinos en el grafo de píxeles (`gnn.knn_k`), configurable.
- **Fecha más cercana.** Sentinel-2 solo pasa sobre la laguna cada pocos días, así
  que la mayoría de fechas no tienen producto. Si la fecha pedida no tiene
  adquisición, el pipeline busca la **más cercana** en `±search_window_days`
  (`config.yaml`, por defecto 7) y la usa automáticamente; la web indica qué fecha
  se utilizó realmente.
- **Filtro de nubes.** Si se proporcionan credenciales **CDS** (en el formulario
  web o como `CDS_ID`/`CDS_SECRET` en CLI), la búsqueda consulta el catálogo y
  elige la fecha **más cercana con nubosidad ≤ `max_cloud_cover`** (`config.yaml`,
  por defecto 20 %); si ninguna baja del umbral, usa la menos lejana disponible y
  avisa. Sin credenciales CDS el filtro se ignora y solo cuenta la cercanía (el
  listado S3 no incluye nubosidad). La web muestra el % de nubes de la fecha usada.

## Solución de problemas

- **El paso de descarga se queda colgado / *timeout* contra
  `eodata.dataspace.copernicus.eu`** aunque el host sí llegue al endpoint: es un
  *blackhole* de MTU en la red por defecto de Docker (el *handshake* TLS se queda a
  "0 bytes received"). Crea una red con MTU reducida y ejecuta el contenedor en
  ella:
  ```bash
  docker network create --opt com.docker.network.driver.mtu=1280 lowmtu
  docker run -it --rm --network lowmtu -p 8000:8000 \
    -v "$(pwd)/gnn_models":/app/gnn_models \
    -v "$(pwd)/data/Chl_Maps":/app/data/Chl_Maps \
    chlwebapp
  ```
  Alternativa permanente: `/etc/docker/daemon.json` con `{"mtu": 1280}` y
  `sudo systemctl restart docker` (afecta a la red por defecto, sin `--network`).
