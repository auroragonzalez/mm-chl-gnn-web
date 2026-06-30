"""
webapp.py — Interfaz web minima para el pipeline de mapas de clorofila.

Envuelve `run_pipeline.py` (descarga Sentinel-2 -> SNAP C2RCC -> prediccion ->
TIFF -> plots/GIF) detras de un formulario web. El usuario solo elige una fecha;
la web lanza el pipeline en segundo plano y, al terminar, muestra los mapas
generados en `map_dir` (config.yaml) y permite descargar los GeoTIFF.

Las credenciales S3 de Copernicus NO se leen de un .env: el usuario las introduce
en el formulario web y se inyectan solo en el subproceso de esa consulta (efimeras,
en memoria; nunca a disco ni al log).

Arranque (dentro del contenedor):
    uvicorn webapp:app --host 0.0.0.0 --port 8000
o simplemente:
    python3 webapp.py

Como el pipeline tarda minutos, cada peticion crea un "job" que corre en un hilo;
la pagina del job se autorrefresca y muestra el log en vivo hasta que aparecen los
mapas.
"""

import os
import re
import glob
import time
import uuid
import threading
import subprocess
from datetime import date as _date, timedelta

import yaml
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

MAP_DIR = CFG.get("map_dir", "/app/data/Chl_Maps/")
# Profundidades esperadas (un mapa "completo" tiene las 4) y ventana de busqueda
# de cache, que reutiliza la misma ventana que la seleccion de fecha del pipeline.
DEPTHS = ["0_1", "1_2", "2_3", "3_4"]
SEARCH_WINDOW = int(CFG.get("search_window_days", 0))

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RESOLVED_RE = re.compile(r"RESOLVED_DATE:\s*(\d{4}-\d{2}-\d{2})")
RESOLVED_CLOUD_RE = re.compile(r"RESOLVED_CLOUD:\s*(\d+(?:\.\d+)?)")
INFER_TIME_RE = re.compile(r"INFER_TIME_S:\s*(\d+(?:\.\d+)?)")
TOTAL_TIME_RE = re.compile(r"TOTAL_TIME_S:\s*(\d+(?:\.\d+)?)")
MAP_VALID_RE = re.compile(r"MAP_VALID:\s*(true|false)", re.IGNORECASE)

# Estado de los jobs en memoria: {job_id: {date, status, log, returncode}}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="Mar Menor — Chlorophyll maps")


# --------------------------------------------------------------------------- #
# Ejecucion del pipeline en segundo plano
# --------------------------------------------------------------------------- #
def _run_pipeline(job_id: str, fecha: str, creds: dict[str, str]) -> None:
    """Lanza run_pipeline.py y vuelca el log al estado del job.

    Las credenciales (`creds`) se inyectan SOLO en el entorno de este subproceso;
    nunca se escriben a disco, ni se guardan en el estado del job, ni se registran
    en el log (el comando que se imprime no las contiene).
    """
    cmd = ["python3", "run_pipeline.py", "--date", fecha]
    # Copia del entorno + credenciales de esta peticion (efimeras, en memoria).
    env = {**os.environ, **{k: v for k, v in creds.items() if v}}
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["log"] = f"$ {' '.join(cmd)}\n"
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:                      # streaming del log
            with JOBS_LOCK:
                JOBS[job_id]["log"] += line
                # El fetcher anuncia la fecha realmente usada (puede diferir de la
                # solicitada si se eligio la adquisicion mas cercana).
                m = RESOLVED_RE.search(line)
                if m:
                    JOBS[job_id]["resolved_date"] = m.group(1)
                mc = RESOLVED_CLOUD_RE.search(line)
                if mc:
                    JOBS[job_id]["resolved_cloud"] = mc.group(1)
                mt = INFER_TIME_RE.search(line)
                if mt:
                    JOBS[job_id]["infer_time"] = mt.group(1)
                mtt = TOTAL_TIME_RE.search(line)
                if mtt:
                    JOBS[job_id]["total_time"] = mtt.group(1)
                mv = MAP_VALID_RE.search(line)
                if mv:
                    JOBS[job_id]["map_valid"] = (mv.group(1).lower() == "true")
        proc.wait()
        with JOBS_LOCK:
            JOBS[job_id]["returncode"] = proc.returncode
            JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as exc:                           # noqa: BLE001
        with JOBS_LOCK:
            JOBS[job_id]["log"] += f"\n[EXCEPCION] {exc}\n"
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["returncode"] = -1


def _maps_for_date(fecha: str) -> dict[str, list[str]]:
    """Devuelve los ficheros generados para una fecha, separados por tipo."""
    patt = os.path.join(MAP_DIR, f"*{fecha}*")
    files = sorted(os.path.basename(p) for p in glob.glob(patt))
    return {
        "images": [f for f in files if f.lower().endswith((".png", ".gif"))],
        "tiffs":  [f for f in files if f.lower().endswith((".tif", ".tiff"))],
    }


def _is_cached(fecha: str) -> bool:
    """True si ya estan en disco los 4 GeoTIFF (mapa completo) de esa fecha."""
    return all(
        os.path.isfile(os.path.join(MAP_DIR, f"{fecha}_chl_map_{d}.tif"))
        for d in DEPTHS
    )


def _depth_is_flat(path: str, tol: float = 1e-4):
    """True si el mapa no tiene variacion real (un solo color); None si no se lee."""
    try:
        import numpy as np
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read(1)
    except Exception:                              # noqa: BLE001
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return True
    return float(np.ptp(finite)) <= tol


def _maps_valid_on_disk(fecha: str):
    """Valida los GeoTIFF en disco para una fecha.

    Devuelve True si al menos un mapa de profundidad tiene variacion real,
    False si TODOS son de un solo color (todos los pixeles iguales), y None si
    no se pudo leer ninguno (validez desconocida -> no se avisa)."""
    flats = [
        _depth_is_flat(os.path.join(MAP_DIR, f"{fecha}_chl_map_{d}.tif"))
        for d in DEPTHS
    ]
    known = [c for c in flats if c is not None]
    if not known:
        return None
    return any(c is False for c in known)


def _find_cached_date(requested: str, window: int) -> str | None:
    """Busca un resultado ya calculado para la fecha pedida o, si no, para la mas
    cercana dentro de +-window dias (mismo criterio que la seleccion del pipeline).
    Devuelve la fecha encontrada (YYYY-MM-DD) o None. Orden: 0, -1, +1, -2, +2, ..."""
    try:
        d0 = _date.fromisoformat(requested)
    except ValueError:
        return None
    candidates = [d0]
    for delta in range(1, max(0, window) + 1):
        candidates.append(d0 - timedelta(days=delta))
        candidates.append(d0 + timedelta(days=delta))
    for c in candidates:
        cs = c.isoformat()
        if _is_cached(cs):
            return cs
    return None


# --------------------------------------------------------------------------- #
# Plantillas HTML (inline, sin dependencias extra)
# --------------------------------------------------------------------------- #
def _page(body: str, refresh: int | None = None) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">{meta}
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mar Menor — Chlorophyll maps</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1b2733}}
 h1{{font-size:1.4rem}} a{{color:#0b6}} .card{{border:1px solid #dde;border-radius:10px;padding:1rem;margin:1rem 0}}
 input,button{{font-size:1rem;padding:.5rem}} button{{background:#0a7;color:#fff;border:0;border-radius:8px;cursor:pointer}}
 pre{{background:#0d1117;color:#c9d1d9;padding:1rem;border-radius:8px;overflow:auto;max-height:340px;font-size:.8rem}}
 .badge{{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.8rem}}
 .run{{background:#fe8;color:#653}} .done{{background:#bf8;color:#252}} .err{{background:#f99;color:#600}}
 img{{max-width:100%;border:1px solid #dde;border-radius:8px;margin:.4rem 0}}
</style></head><body>{body}</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    today = _date.today().isoformat()
    body = f"""
    <h1>Mar Menor · chlorophyll-a map</h1>
    <div class="card" style="border-color:#9bd;background:#f6faff">
      <h2 style="font-size:1.1rem;margin-top:0">Before you start: how to get your credentials</h2>
      <p style="margin:.4rem 0"><strong>Copernicus S3 access key and secret key</strong></p>
      <ol style="margin:.2rem 0 .8rem;padding-left:1.2rem;line-height:1.5">
        <li>Register in the Copernicus system at
          <a href="https://dataspace.copernicus.eu/" target="_blank" rel="noopener">dataspace.copernicus.eu</a>.</li>
        <li>Log in to your Copernicus account.</li>
        <li>Generate an S3 access key and secret key at
          <a href="https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials" target="_blank" rel="noopener">the S3 keys manager</a>.</li>
        <li><strong>The secret can only be viewed once</strong>, so make note of it.</li>
      </ol>
      <p style="margin:.4rem 0"><strong>CDS credentials (optional, for cloud filtering)</strong></p>
      <ol style="margin:.2rem 0 .8rem;padding-left:1.2rem;line-height:1.5">
        <li>Register an OAuth client in your
          <a href="https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings" target="_blank" rel="noopener">account settings</a>.</li>
        <li>This gives you a client id and a client secret. As before,
          <strong>the secret can only be viewed once</strong>, so make note of it.</li>
      </ol>
      <p style="font-size:.85rem;color:#445;margin:.4rem 0 0">
        More details:
        <a href="https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html" target="_blank" rel="noopener">Sentinel Hub authentication documentation</a>.</p>
    </div>
    <div class="card">
      <p>Pick a date with a Sentinel-2 overpass over the lagoon. The system downloads
      the product, applies the C2RCC atmospheric correction and generates the Chl-a maps.</p>
      <form action="/predict" method="post" autocomplete="off">
        <p><label>Date: <input type="date" name="fecha" value="{today}" required></label></p>
        <fieldset style="border:1px solid #dde;border-radius:8px;padding:.6rem 1rem">
          <legend style="font-size:.85rem;color:#445">Copernicus credentials (S3)</legend>
          <p style="color:#667;font-size:.8rem;margin:.2rem 0 .6rem">
            These are not stored on the server: they are used only for this query and discarded when it finishes.
            They are only needed if the date has not been computed yet (if it has, the result is retrieved from disk).</p>
          <p><label>S3 access key<br>
            <input type="password" id="s3a" name="s3_access_key" style="width:100%"></label></p>
          <p><label>S3 secret key<br>
            <input type="password" id="s3s" name="s3_secret_key" style="width:100%"></label></p>
          <p style="font-size:.85rem"><label>
            <input type="checkbox" id="remember"> Remember in this browser (never sent to the server to be stored)</label></p>
        </fieldset>
        <fieldset style="border:1px solid #dde;border-radius:8px;padding:.6rem 1rem;margin-top:.6rem">
          <legend style="font-size:.85rem;color:#445">Cloud filter — CDS credentials (optional)</legend>
          <p style="color:#667;font-size:.8rem;margin:.2rem 0 .6rem">
            If you fill these in, the nearest date with low cloud cover is chosen.
            If you leave them empty, only proximity is used. These are not stored on the server either.</p>
          <p><label>CDS client id<br>
            <input type="password" id="cdsid" name="cds_id" style="width:100%"></label></p>
          <p><label>CDS client secret<br>
            <input type="password" id="cdssec" name="cds_secret" style="width:100%"></label></p>
        </fieldset>
        <p><button type="submit">Generate map</button></p>
      </form>
      <p style="color:#667;font-size:.85rem">The process takes several minutes (download + SNAP).
      It is a good idea to first check that the date has no clouds over the lagoon.</p>
    </div>
    <script>
      // Optional convenience: the keys are stored ONLY in the user's browser
      // (localStorage), never on the server.
      (function() {{
        var a = document.getElementById('s3a'), s = document.getElementById('s3s'),
            r = document.getElementById('remember'),
            ci = document.getElementById('cdsid'), cs = document.getElementById('cdssec');
        try {{
          var sa = localStorage.getItem('mm_s3a'), ss = localStorage.getItem('mm_s3s'),
              sci = localStorage.getItem('mm_cdsid'), scs = localStorage.getItem('mm_cdssec');
          if (sa) {{ a.value = sa; }} if (ss) {{ s.value = ss; }}
          if (sci) {{ ci.value = sci; }} if (scs) {{ cs.value = scs; }}
          if (sa || ss || sci || scs) {{ r.checked = true; }}
        }} catch (e) {{}}
        document.querySelector('form').addEventListener('submit', function() {{
          try {{
            if (r.checked) {{
              localStorage.setItem('mm_s3a', a.value); localStorage.setItem('mm_s3s', s.value);
              localStorage.setItem('mm_cdsid', ci.value); localStorage.setItem('mm_cdssec', cs.value);
            }} else {{
              localStorage.removeItem('mm_s3a'); localStorage.removeItem('mm_s3s');
              localStorage.removeItem('mm_cdsid'); localStorage.removeItem('mm_cdssec');
            }}
          }} catch (e) {{}}
        }});
      }})();
    </script>"""
    return _page(body)


@app.post("/predict")
def predict(
    fecha: str = Form(...),
    s3_access_key: str = Form(...),
    s3_secret_key: str = Form(...),
    cds_id: str = Form(""),
    cds_secret: str = Form(""),
) -> RedirectResponse:
    if not DATE_RE.match(fecha):
        raise HTTPException(400, "Invalid date format (expected YYYY-MM-DD).")

    # 1) Cache: si ya existe el resultado de esa fecha (o de una cercana en
    # +-search_window_days) en disco, se recupera SIN relanzar el pipeline ni
    # necesitar credenciales.
    cached = _find_cached_date(fecha, SEARCH_WINDOW)
    if cached:
        job_id = uuid.uuid4().hex[:8]
        msg = (f"Result already available on disk for {cached}"
               + ("" if cached == fecha else f" (the nearest one to {fecha})")
               + "; retrieved without rerunning the pipeline.\n")
        with JOBS_LOCK:
            JOBS[job_id] = {"date": fecha, "resolved_date": cached, "resolved_cloud": None,
                            "infer_time": None, "total_time": None, "status": "done",
                            "log": msg, "returncode": 0, "from_cache": True,
                            "map_valid": None}
        return RedirectResponse(url=f"/job/{job_id}", status_code=303)

    # 2) Sin cache: hay que descargar -> se requieren credenciales S3.
    if not s3_access_key.strip() or not s3_secret_key.strip():
        raise HTTPException(400, "No saved result exists for that date, and the Copernicus "
                                 "S3 credentials needed to generate it are missing.")
    creds = {
        "S3_ACCESS_KEY": s3_access_key.strip(),
        "S3_SECRET_KEY": s3_secret_key.strip(),
    }
    # CDS opcionales: solo habilitan el filtro de nubes si se proporcionan.
    if cds_id.strip() and cds_secret.strip():
        creds["CDS_ID"] = cds_id.strip()
        creds["CDS_SECRET"] = cds_secret.strip()
    job_id = uuid.uuid4().hex[:8]
    with JOBS_LOCK:
        # OJO: el estado del job NO guarda las credenciales (solo metadatos/log).
        JOBS[job_id] = {"date": fecha, "resolved_date": None, "resolved_cloud": None,
                        "infer_time": None, "total_time": None, "status": "queued",
                        "log": "", "returncode": None, "from_cache": False,
                        "map_valid": None, "start_ts": time.time()}
    threading.Thread(target=_run_pipeline, args=(job_id, fecha, creds), daemon=True).start()
    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> str:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")

    status = job["status"]
    fecha = job["date"]
    resolved = job.get("resolved_date")
    cloud = job.get("resolved_cloud")
    infer_time = job.get("infer_time")
    total_time = job.get("total_time")
    # Fecha con la que se nombran los mapas: la resuelta si existe, si no la pedida.
    map_date = resolved or fecha
    title_date = fecha if not resolved or resolved == fecha else f"{resolved}"
    cloud_txt = f" · clouds {cloud}%" if cloud is not None else ""
    total_txt = f" · total {total_time}s" if total_time is not None else ""
    badge = {"running": "run", "queued": "run", "done": "done", "error": "err"}.get(status, "run")
    status_label = {"running": "running", "queued": "queued",
                    "done": "done", "error": "error"}.get(status, status)
    head = f'<h1>Date {title_date}{cloud_txt}{total_txt} · <span class="badge {badge}">{status_label}</span></h1>' \
           f'<p><a href="/">&larr; new query</a></p>'

    if total_time is not None:
        try:
            mins = f" (~{float(total_time) / 60:.1f} min)"
        except (TypeError, ValueError):
            mins = ""
        detalle = f" · GNN inference: {infer_time} s" if infer_time is not None else ""
        head += ('<div class="card" style="border-color:#9bd;background:#f6faff">'
                 f'⏱️ <strong>Total pipeline time: {total_time} s</strong>{mins}{detalle}.</div>')

    if resolved and resolved != fecha:
        extra = f" Estimated cloud cover: <strong>{cloud}%</strong>." if cloud is not None else ""
        head += (f'<div class="card" style="border-color:#fc8;background:#fffdf6">'
                 f'The requested date <strong>{fecha}</strong> was not suitable; '
                 f'the nearest suitable one was used: <strong>{resolved}</strong>.{extra}</div>')

    if job.get("from_cache"):
        head += ('<div class="card" style="border-color:#9c9;background:#f6fff8">'
                 'Result <strong>retrieved from a previous computation</strong> on disk '
                 '(the pipeline was not rerun).</div>')

    if status in ("queued", "running"):
        # Tiempo ya transcurrido en el servidor; el JS sigue contando desde ahi
        # (cada refresco de 4s lo recalibra con el valor del servidor).
        elapsed = 0
        if job.get("start_ts"):
            elapsed = int(time.time() - job["start_ts"])
        body = head + (
            '<div class="card">'
            '<p style="margin:.2rem 0">Processing… (this page refreshes automatically)</p>'
            '<p style="font-size:1.6rem;margin:.4rem 0">⏱️ '
            f'<strong id="timer" data-elapsed="{elapsed}">00:00</strong>'
            ' <span style="font-size:.9rem;color:#667">elapsed</span></p>'
            '<p style="font-size:.85rem;color:#667;margin:.2rem 0">'
            'This usually takes several minutes (download + atmospheric correction).</p>'
            '</div>'
            f'<pre>{_escape(job["log"])[-6000:]}</pre>'
            '<script>'
            '(function(){'
            ' var el=document.getElementById("timer");'
            ' if(!el)return;'
            ' var t=parseInt(el.getAttribute("data-elapsed"),10)||0;'
            ' function fmt(s){var m=Math.floor(s/60),x=s%60;'
            '  return (m<10?"0":"")+m+":"+(x<10?"0":"")+x;}'
            ' el.textContent=fmt(t);'
            ' setInterval(function(){t++;el.textContent=fmt(t);},1000);'
            '})();'
            '</script>'
        )
        return _page(body, refresh=4)

    # terminado (done o error)
    maps = _maps_for_date(map_date)
    if status == "error":
        head += '<div class="card" style="border-color:#f99">The pipeline finished with an error. ' \
                'Check the log (it may be a missing product/clouds or credentials).</div>'

    # Validez del resultado: si todos los mapas son de un solo color, avisar.
    # Para jobs en vivo viene del marcador; para resultados cacheados se mira en disco.
    if status == "done":
        map_valid = job.get("map_valid")
        if map_valid is None:
            map_valid = _maps_valid_on_disk(map_date)
        if map_valid is False:
            head += ('<div class="card" style="border-color:#f0ad4e;background:#fff8ec">'
                     '⚠️ <strong>Low-quality result.</strong> The generated maps have a '
                     'single uniform color (all pixels are equal), which means the source '
                     'Sentinel-2 image for this date did not have enough quality for the '
                     'pipeline to work well. <strong>Please try a different date.</strong></div>')

    imgs = "".join(
        f'<div><strong>{f}</strong><br><img src="/maps/{f}" alt="{f}"></div>'
        for f in maps["images"]
    ) or "<p>No images were generated for this date.</p>"

    tifs = "".join(
        f'<li><a href="/maps/{f}">{f}</a></li>' for f in maps["tiffs"]
    )
    tif_block = f"<div class='card'><strong>Downloadable GeoTIFFs</strong><ul>{tifs}</ul></div>" if tifs else ""

    body = head + f"<div class='card'>{imgs}</div>{tif_block}" \
                  f"<details><summary>View log</summary><pre>{_escape(job['log'])[-12000:]}</pre></details>"
    return _page(body)


@app.get("/maps/{filename}")
def serve_map(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid file name.")
    path = os.path.join(MAP_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "File not found.")
    return FileResponse(path)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    import uvicorn
    # Usar el puerto que inyecte la plataforma (PORT o SERVER_PORT); si no, 8000.
    # Siempre en 0.0.0.0 para que sea accesible desde fuera del contenedor.
    port = int(os.getenv("PORT") or os.getenv("SERVER_PORT") or "8000")
    print(f"[webapp] escuchando en 0.0.0.0:{port}")
    uvicorn.run("webapp:app", host="0.0.0.0", port=port)
