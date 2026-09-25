# BioBehaviour — Análisis de comportamiento animal

Aplicación de escritorio en Python que unifica tres herramientas de análisis
de video en una sola ventana con pestañas:

1. **Pulso cardíaco** (daphnias): estima la frecuencia cardíaca (BPM) a
   partir de un video y una región de interés elegida por el usuario.
2. **Rastreo de movimiento**: sigue a la daphnia en el video y genera datos
   de trayectoria, distancia, velocidad y giros.
3. **Mapa de calor**: genera un mapa de ocupación del animal a partir de un
   video (o de una carpeta completa, en lote).

Las tres pestañas comparten la misma interfaz: panel de controles a la
izquierda y visor/resultados a la derecha. Cada análisis corre en su propio
hilo, por lo que se puede cambiar de pestaña mientras algo se procesa.

## Requisitos

- Python 3.9 o superior (probado con Python 3.10 en Windows).
- Los paquetes de `requirements.txt`.

## Instalación

Abra una terminal (PowerShell) en esta carpeta y ejecute para crear un entorno que no interfiera con sus demás proyectos:

```powershell
python -m venv .bbvenv
.bbvenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Después de instalar, puede abrir la aplicación de dos formas:

- O en la terminal: `python main.py`

> Para una instalación con versiones exactas ya probadas use
> `requirements-lock.txt` en lugar de `requirements.txt`.

## Uso

### Pestaña 1 · Pulso cardíaco

1. **Cargar video** (.mp4, .avi, .mov, .mkv).
2. Arrastre el ratón sobre el video para enmarcar el tórax de la daphnia
   (la región donde late el corazón).
3. Pulse **▶ Analizar pulso**. Verá el BPM estimado y la señal cardíaca.
   - *Análisis en tiempo real*: si está activado, dibuja el recuadro sobre
     cada frame (más lento). Desactivado es el modo acelerado recomendado.
4. Escriba el nombre de la muestra y pulse **Exportar resultados**: genera
   `nombre.csv` (21 columnas: BPM, señal, picos, metadatos del análisis) y
   `nombre.png` (gráfica de la señal).

### Pestaña 2 · Rastreo de movimiento

1. **Cargar video** (.mp4, .avi, .mov).
2. Ajuste el *umbral de movimiento* (px que cuentan como desplazamiento
   real, no movimiento de apéndices) y la *sensibilidad de detección*.
3. Opcional: active la **calibración** para obtener distancia y velocidad en
   unidades reales (p. ej. 100 píxeles = 1 mm).
4. Elija la carpeta de salida (se sugiere `rastreo_salidas` junto al video).
5. Pulse **▶ Iniciar análisis**: verá el video con la trayectoria superpuesta.

Genera 5 archivos por video (con el prefijo del nombre del video):

| Archivo | Contenido |
|---|---|
| `*_datos.csv` | Trayectoria frame a frame (x, y, tiempo, velocidad…) |
| `*_resumen.txt` | Estadísticas: distancias, velocidades, giros |
| `*_larga_exposicion.png` | Estela del movimiento (larga exposición) |
| `*_mapa_calor.png` | Mapa de calor de ocupación |
| `*_grafica_velocidad.png` | Velocidad a lo largo del tiempo |

### Pestaña 3 · Mapa de calor

1. Elija **un solo video** o **una carpeta completa** (procesa en lote todos
   los .mp4/.avi/.mkv/.mov/.wmv).
2. Ajuste los parámetros si lo necesita:
   - *Muestrear fondo cada N frames* y *máx. frames para fondo*: construyen
     el fondo (mediana) para restar.
   - *Umbral de detección*: sensibilidad de la diferencia contra el fondo.
   - *Área mínima*: descarta contornos pequeños (ruido).
   - *Kernel de blur*: suavizado previo (0 = sin suavizado).
   - *Acumular máscara completa*: suma el cuerpo completo; si está
     desactivado se suma solo el centroide.
3. Elija la carpeta de salida y pulse **▶ Procesar mapa de calor**.

Por cada video genera `*_heatmap.npz` (datos numéricos: heat en frames y en
segundos, fps y dimensiones) y `*_heatmap.png` (mapa de calor). La vista
previa del resultado se muestra en el visor y la bitácora inferior registra
todo lo generado.

## Solución de problemas

- **Error "faltan dependencias"**: ejecute `pip install -r requirements.txt`.
- **El doble clic en main.pyw no hace nada**: revise que `python` esté
  instalado y asociado (instale desde python.org marcando *Add to PATH*), o
  abra la app con `python main.py` para ver el error en la consola.
- **Videos que no abren**: pruebe con .mp4 (H.264) o .avi; algunos códecs
  propietarios no están disponibles en OpenCV.

## Notas de la integración

- Interfaz construida con **PySide6** (Qt 6) para toda la aplicación; no se
  mezclan bindings de Qt, lo que evita conflictos y cuelgues.
- El detector de la pestaña de rastreo corrige dos defectos del original:
  se elimina un `BackgroundSubtractorMOG2` que se recreaba en cada frame
  (invalidando la máscara adaptativa) y una llamada incorrecta a
  `imutils.grab_contours`. El resto del algoritmo (umbral adaptativo/Otsu,
  morfología, contorno mayor) es el documentado en el código original.
- Matplotlib se usa únicamente con el backend `Agg`: las gráficas se guardan
  como PNG, nunca se abren ventanas ni se necesita consola.
- Los formatos de salida (CSV de 21 columnas, NPZ, nombres de archivo) se
  conservan respecto a las aplicaciones originales.
