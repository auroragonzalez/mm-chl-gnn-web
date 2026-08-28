#!/bin/bash

# === Argumentos ===
FECHA=$1
INPUT_DIR=$2
OUTPUT_DIR=$3

# === Configuración del grafo y GPT ===
GRAPH_XML="/app/fetch/snap_graph_application.xml"
TEMPLATE_PARAMS="/app/fetch/snap_graph_application.properties"
GPT="/opt/snap/bin/gpt"
OUTPUT_FORMAT="tif"

# === Validaciones ===
if [ -z "$FECHA" ] || [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Uso: bash snap_batch_application.sh <fecha> <input_dir> <output_dir>"
  exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
  echo "No se encuentra el directorio de entrada: $INPUT_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# === Mostrar información ===
echo "=== Ejecutando SNAP para la fecha: $FECHA ==="
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Usando GPT en: $GPT"

# === Filtrado de productos SAFE ===
FILTER_DATE=$(echo "$FECHA" | sed 's/-//g')  # Convierte 2022-07-14 -> 20220714
FILTER_DATES=($FILTER_DATE)



if [[ "${#FILTER_DATES[@]}" -gt 0 ]]; then
    echo "Procesando las fechas: ${FILTER_DATES[*]}"
else
    echo "Procesando todos los archivos disponibles"
fi


for input_file in "$INPUT_DIR"/*.SAFE.zip; do
    # Extraer identificadores del nombre
    filename=$(basename "$input_file")
    datecode=$(echo "$filename" | cut -d'_' -f3 | cut -c1-8)
    base_name="${filename%%.SAFE.zip}"

    # Filtrar si se especificaron fechas
    if [[ "${#FILTER_DATES[@]}" -gt 0 && ! " ${FILTER_DATES[*]} " =~ " $datecode " ]]; then
        echo "Saltando $filename (fecha $datecode no en filtro)"
        continue
    fi

    # Leer el tipo de red desde la plantilla
    net=$(grep "^netSet=" "$TEMPLATE_PARAMS" | cut -d= -f2 | tr -d '[:space:]')
    if [[ "$net" == "C2X-COMPLEX-Nets" ]]; then
        suffix="C2XComplexNets"
    elif [[ "$net" == "C2X-Nets" ]]; then
        suffix="C2XNets"
    else
        suffix="C2RCCNets"
    fi

    output_file="$OUTPUT_DIR/${base_name}_${suffix}_10m.${OUTPUT_FORMAT}"

    # Crear archivo de parámetros específico
    param_file="/tmp/params_${datecode}.params"
    sed "s|inputFile=.*|inputFile=$input_file|; s|outputFile=.*|outputFile=$output_file|" "$TEMPLATE_PARAMS" > "$param_file"

    # Ejecutar gpt
    #  -c  limita la cache de tiles (debe ser < -Xmx de gpt.vmoptions, fijado en el Dockerfile)
    #  -q  limita el paralelismo
    # Ambos acotan el pico de memoria y hacen el procesado mas determinista:
    # sin ellos la correccion atmosferica puede degenerar y salir un mapa de un solo color.
    echo "Procesando $filename → $output_file"
    "$GPT" "$GRAPH_XML" -p "$param_file" -c "${SNAP_TILE_CACHE:-2G}" -q "${SNAP_PARALLELISM:-4}"
    gpt_status=$?
    if [ "$gpt_status" -ne 0 ]; then
        echo "ERROR: gpt fallo para $filename (codigo $gpt_status). Abortando." >&2
        exit 1
    fi

    # La correccion atmosferica debe haber generado un TIFF no vacio.
    if [ ! -s "$output_file" ]; then
        echo "ERROR: no se genero el TIFF de salida $output_file. Abortando." >&2
        exit 1
    fi
done

