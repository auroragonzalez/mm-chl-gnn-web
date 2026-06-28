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

# Estado de los jobs en memoria: {job_id: {date, status, log, returncode}}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="Mar Menor — Mapas de clorofila")


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
<html lang="es"><head><meta charset="utf-8">{meta}
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mar Menor — Mapas de clorofila</title>
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
    <h1>Mar Menor · mapa de clorofila-a</h1>
    <div class="card">
      <p>Elige una fecha con paso de Sentinel-2 sobre la laguna. El sistema descarga
      el producto, aplica la correccion atmosferica C2RCC y genera los mapas de Chl-a.</p>
      <form action="/predict" method="post" autocomplete="off">
        <p><label>Fecha: <input type="date" name="fecha" value="{today}" required></label></p>
        <fieldset style="border:1px solid #dde;border-radius:8px;padding:.6rem 1rem">
          <legend style="font-size:.85rem;color:#445">Credenciales Copernicus (S3)</legend>
          <p style="color:#667;font-size:.8rem;margin:.2rem 0 .6rem">
            No se guardan en el servidor: se usan solo para esta consulta y se descartan al terminar.
            Solo son necesarias si la fecha no está ya calculada (si lo está, se recupera de disco).</p>
          <p><label>S3 access key<br>
            <input type="password" id="s3a" name="s3_access_key" style="width:100%"></label></p>
          <p><label>S3 secret key<br>
            <input type="password" id="s3s" name="s3_secret_key" style="width:100%"></label></p>
          <p style="font-size:.85rem"><label>
            <input type="checkbox" id="remember"> Recordar en este navegador (no se envia al servidor para almacenarse)</label></p>
        </fieldset>
        <fieldset style="border:1px solid #dde;border-radius:8px;padding:.6rem 1rem;margin-top:.6rem">
          <legend style="font-size:.85rem;color:#445">Filtro de nubes — credenciales CDS (opcional)</legend>
          <p style="color:#667;font-size:.8rem;margin:.2rem 0 .6rem">
            Si las rellenas, se elige la fecha más cercana con poca nubosidad
            (umbral en <code>config.yaml</code>). Si las dejas vacías, se usa solo la
            cercanía. Tampoco se guardan en el servidor.</p>
          <p><label>CDS client id<br>
            <input type="password" id="cdsid" name="cds_id" style="width:100%"></label></p>
          <p><label>CDS client secret<br>
            <input type="password" id="cdssec" name="cds_secret" style="width:100%"></label></p>
        </fieldset>
        <p><button type="submit">Generar mapa</button></p>
      </form>
      <p style="color:#667;font-size:.85rem">El proceso tarda varios minutos (descarga + SNAP).
      Conviene comprobar antes que la fecha no tiene nubes sobre la laguna.</p>
    </div>
    <script>
      // Conveniencia opcional: las claves se guardan SOLO en el navegador del
      // usuario (localStorage), nunca en el servidor.
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
        raise HTTPException(400, "Formato de fecha invalido (se espera YYYY-MM-DD).")

    # 1) Cache: si ya existe el resultado de esa fecha (o de una cercana en
    # +-search_window_days) en disco, se recupera SIN relanzar el pipeline ni
    # necesitar credenciales.
    cached = _find_cached_date(fecha, SEARCH_WINDOW)
    if cached:
        job_id = uuid.uuid4().hex[:8]
        msg = (f"Resultado ya disponible en disco para {cached}"
               + ("" if cached == fecha else f" (la mas cercana a {fecha})")
               + "; recuperado sin relanzar el pipeline.\n")
        with JOBS_LOCK:
            JOBS[job_id] = {"date": fecha, "resolved_date": cached, "resolved_cloud": None,
                            "infer_time": None, "total_time": None, "status": "done",
                            "log": msg, "returncode": 0, "from_cache": True}
        return RedirectResponse(url=f"/job/{job_id}", status_code=303)

    # 2) Sin cache: hay que descargar -> se requieren credenciales S3.
    if not s3_access_key.strip() or not s3_secret_key.strip():
        raise HTTPException(400, "No hay resultado guardado para esa fecha y faltan las "
                                 "credenciales S3 de Copernicus para generarlo.")
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
                        "log": "", "returncode": None, "from_cache": False}
    threading.Thread(target=_run_pipeline, args=(job_id, fecha, creds), daemon=True).start()
    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str) -> str:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado.")

    status = job["status"]
    fecha = job["date"]
    resolved = job.get("resolved_date")
    cloud = job.get("resolved_cloud")
    infer_time = job.get("infer_time")
    total_time = job.get("total_time")
    # Fecha con la que se nombran los mapas: la resuelta si existe, si no la pedida.
    map_date = resolved or fecha
    title_date = fecha if not resolved or resolved == fecha else f"{resolved}"
    cloud_txt = f" · nubes {cloud}%" if cloud is not None else ""
    total_txt = f" · total {total_time}s" if total_time is not None else ""
    badge = {"running": "run", "queued": "run", "done": "done", "error": "err"}.get(status, "run")
    head = f'<h1>Fecha {title_date}{cloud_txt}{total_txt} · <span class="badge {badge}">{status}</span></h1>' \
           f'<p><a href="/">&larr; nueva consulta</a></p>'

    if total_time is not None:
        try:
            mins = f" (~{float(total_time) / 60:.1f} min)"
        except (TypeError, ValueError):
            mins = ""
        detalle = f" · inferencia GNN: {infer_time} s" if infer_time is not None else ""
        head += ('<div class="card" style="border-color:#9bd;background:#f6faff">'
                 f'⏱️ <strong>Tiempo total del pipeline: {total_time} s</strong>{mins}{detalle}. '
                 'Desglose completo por fases en <code>inference_times.csv</code>.</div>')

    if resolved and resolved != fecha:
        extra = f" Nubosidad estimada: <strong>{cloud}%</strong>." if cloud is not None else ""
        head += (f'<div class="card" style="border-color:#fc8;background:#fffdf6">'
                 f'La fecha solicitada <strong>{fecha}</strong> no era idónea; '
                 f'se usó la más cercana adecuada: <strong>{resolved}</strong>.{extra}</div>')

    if job.get("from_cache"):
        head += ('<div class="card" style="border-color:#9c9;background:#f6fff8">'
                 'Resultado <strong>recuperado de cálculos previos</strong> en disco '
                 '(no se relanzó el pipeline).</div>')

    if status in ("queued", "running"):
        body = head + f'<div class="card">Procesando… (esta pagina se actualiza sola)</div>' \
                      f'<pre>{_escape(job["log"])[-6000:]}</pre>'
        return _page(body, refresh=4)

    # terminado (done o error)
    maps = _maps_for_date(map_date)
    if status == "error":
        head += '<div class="card" style="border-color:#f99">El pipeline termino con error. ' \
                'Revisa el log (puede ser falta de producto/nubes o credenciales).</div>'

    imgs = "".join(
        f'<div><strong>{f}</strong><br><img src="/maps/{f}" alt="{f}"></div>'
        for f in maps["images"]
    ) or "<p>No se generaron imagenes para esta fecha.</p>"

    tifs = "".join(
        f'<li><a href="/maps/{f}">{f}</a></li>' for f in maps["tiffs"]
    )
    tif_block = f"<div class='card'><strong>GeoTIFF descargables</strong><ul>{tifs}</ul></div>" if tifs else ""

    body = head + f"<div class='card'>{imgs}</div>{tif_block}" \
                  f"<details><summary>Ver log</summary><pre>{_escape(job['log'])[-12000:]}</pre></details>"
    return _page(body)


@app.get("/maps/{filename}")
def serve_map(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Nombre de fichero invalido.")
    path = os.path.join(MAP_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Fichero no encontrado.")
    return FileResponse(path)


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
