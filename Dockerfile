FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH="/opt/snap/bin:/usr/local/bin:${PATH}"

# === Dependencias de sistema ===
RUN apt-get update && apt-get install -y \
    wget curl unzip build-essential \
    libssl-dev libffi-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev \
    openjdk-11-jdk gdal-bin tk-dev liblzma-dev \
    && apt-get clean

# === Instalar Python 3.11.0 desde fuente ===
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz && \
    tar -xf Python-3.11.0.tgz && cd Python-3.11.0 && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && make altinstall && \
    ln -sf /usr/local/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3 && \
    cd .. && rm -rf Python-3.11.0*

# === Instalar SNAP 12.0.0 (correccion atmosferica C2RCC) ===
RUN wget -O /tmp/snap_installer_12.sh "https://download.esa.int/step/snap/12.0/installers/esa-snap_all_linux-12.0.0.sh" && \
    chmod +x /tmp/snap_installer_12.sh && \
    /tmp/snap_installer_12.sh -q -dir /opt/snap && \
    rm /tmp/snap_installer_12.sh

# === Proyecto ===
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY fetch/ ./fetch/
COPY models/ ./models/
COPY gnn/ ./gnn/
COPY run_pipeline.py .
COPY webapp.py .
COPY config.yaml .
COPY check_dates.py .

# Carpetas de trabajo / resultados
RUN mkdir -p /app/data/Chl_Maps /app/data/preds /app/data/processed /app/data/Copernicus /app/gnn_models
VOLUME ["/app/data/Chl_Maps", "/app/gnn_models"]

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SNAP_JAVA_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED"
RUN mkdir -p /root/.snap

EXPOSE 8000

# Por defecto arranca la web. Para la CLI clasica:
#   docker run ... chlwebapp python3 run_pipeline.py --date 2022-07-14
CMD ["python3", "webapp.py"]
