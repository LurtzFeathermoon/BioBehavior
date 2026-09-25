"""Módulo de generación de salidas.

Produce:
    1. CSV con la serie temporal (tiempo, x, y, velocidad, estado, sentido).
    2. Imagen de larga exposición (composición de todas las posiciones).
    3. Mapa de calor de densidad de ubicaciones.
    4. Gráfica matplotlib de tiempo vs velocidad.
    5. Un resumen de texto con las métricas agregadas.
"""

from __future__ import annotations

import csv
import os
from typing import List, Optional

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")  # backend sin GUI para guardar archivos
import matplotlib.pyplot as plt

from .analysis import ResultadoAnalisis


def _asegurar_carpeta(carpeta: str) -> None:
    os.makedirs(carpeta, exist_ok=True)


def exportar_csv(resultado: ResultadoAnalisis, ruta: str) -> str:
    """Escribe el CSV con la serie temporal del análisis."""
    _asegurar_carpeta(os.path.dirname(ruta) or ".")
    cal = resultado.calibracion
    unidad_v = cal.unidad_velocidad
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "frame",
                "tiempo_s",
                "posicion_x_px",
                "posicion_y_px",
                "desplazamiento_px",
                "velocidad_px_s",
                f"velocidad_{unidad_v}",
                "estado",
                "sentido_giro",
                "detectado",
            ]
        )
        for p in resultado.puntos:
            w.writerow(
                [
                    p.indice,
                    f"{p.tiempo:.4f}",
                    "" if p.x is None else f"{p.x:.2f}",
                    "" if p.y is None else f"{p.y:.2f}",
                    f"{p.desplazamiento_px:.3f}",
                    f"{p.velocidad_px:.3f}",
                    f"{p.velocidad_real:.5f}",
                    p.estado,
                    p.sentido_giro,
                    "si" if p.detectado else "no",
                ]
            )
    return ruta


def exportar_resumen(resultado: ResultadoAnalisis, ruta: str) -> str:
    """Escribe un resumen de texto con las métricas agregadas."""
    _asegurar_carpeta(os.path.dirname(ruta) or ".")
    cal = resultado.calibracion
    ud = cal.unidad_distancia
    uv = cal.unidad_velocidad
    lineas = [
        "=== RESUMEN DE ANÁLISIS - BioBehaviour ===",
        "",
        f"FPS del video:               {resultado.fps:.2f}",
        f"Frames totales:              {resultado.frames_totales}",
        f"Frames con detección:        {resultado.frames_detectados}",
        f"Tiempo total:                {resultado.tiempo_total:.2f} s",
        f"Tiempo en movimiento:        {resultado.tiempo_movimiento:.2f} s",
        f"Tiempo estático:             {resultado.tiempo_estatico:.2f} s",
        "",
        "--- Distancia y velocidad (píxeles) ---",
        f"Distancia total:             {resultado.distancia_total_px:.2f} px",
        f"Velocidad promedio:          {resultado.velocidad_promedio_px:.2f} px/s",
        f"Velocidad máxima:            {resultado.velocidad_maxima_px:.2f} px/s",
    ]
    if cal.activa:
        lineas += [
            "",
            f"--- Distancia y velocidad ({ud}) ---",
            f"Calibración:                 {cal.pixeles:g} px = {cal.valor_real:g} {cal.unidad}",
            f"Distancia total:             {resultado.distancia_total_real:.4f} {ud}",
            f"Velocidad promedio:          {resultado.velocidad_promedio_real:.4f} {uv}",
            f"Velocidad máxima:            {resultado.velocidad_maxima_real:.4f} {uv}",
        ]
    lineas += [
        "",
        "--- Sentido de giro ---",
        f"Sentido predominante:        {resultado.sentido_predominante}",
        f"Frames horario (CW):         {resultado.giros_cw}",
        f"Frames antihorario (CCW):    {resultado.giros_ccw}",
        "",
    ]
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    return ruta


def generar_larga_exposicion(
    resultado: ResultadoAnalisis,
    ancho: int,
    alto: int,
    ruta: str,
    frame_fondo: Optional[np.ndarray] = None,
) -> str:
    """Genera la imagen de larga exposición (trail de la trayectoria).

    Dibuja un gradiente de color a lo largo del tiempo sobre un fondo
    oscuro (o sobre el último frame si se proporciona).
    """
    _asegurar_carpeta(os.path.dirname(ruta) or ".")
    if frame_fondo is not None:
        lienzo = (frame_fondo.astype(np.float32) * 0.35).astype(np.uint8)
    else:
        lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)

    puntos = [(p.x, p.y) for p in resultado.puntos if p.x is not None]
    n = len(puntos)
    if n >= 2:
        for i in range(1, n):
            x1, y1 = puntos[i - 1]
            x2, y2 = puntos[i]
            t = i / n
            # Gradiente de azul (inicio) a rojo (final) en BGR
            color = (
                int(255 * (1 - t)),  # B
                int(80),             # G
                int(255 * t),        # R
            )
            cv2.line(
                lienzo,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
                cv2.LINE_AA,
            )
    # Marcar inicio y fin
    if puntos:
        x0, y0 = puntos[0]
        xf, yf = puntos[-1]
        cv2.circle(lienzo, (int(x0), int(y0)), 6, (255, 150, 0), -1, cv2.LINE_AA)
        cv2.circle(lienzo, (int(xf), int(yf)), 6, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.imwrite(ruta, lienzo)
    return ruta


def generar_mapa_calor(
    resultado: ResultadoAnalisis,
    ancho: int,
    alto: int,
    ruta: str,
    radio_kernel: int = 25,
) -> str:
    """Genera un mapa de calor de densidad de ubicaciones."""
    _asegurar_carpeta(os.path.dirname(ruta) or ".")
    acumulador = np.zeros((alto, ancho), dtype=np.float32)
    for p in resultado.puntos:
        if p.x is None:
            continue
        x = int(np.clip(p.x, 0, ancho - 1))
        y = int(np.clip(p.y, 0, alto - 1))
        acumulador[y, x] += 1.0

    # Suavizado gaussiano para crear la densidad
    k = radio_kernel if radio_kernel % 2 == 1 else radio_kernel + 1
    densidad = cv2.GaussianBlur(acumulador, (k, k), 0)
    if densidad.max() > 0:
        densidad = densidad / densidad.max()
    densidad_u8 = (densidad * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(densidad_u8, cv2.COLORMAP_JET)
    # Fondo negro donde no hay densidad
    mascara = (densidad_u8 == 0)
    heatmap[mascara] = (0, 0, 0)
    cv2.imwrite(ruta, heatmap)
    return ruta


def generar_grafica_velocidad(
    resultado: ResultadoAnalisis, ruta: str
) -> str:
    """Genera la gráfica matplotlib: tiempo (X) vs velocidad (Y)."""
    _asegurar_carpeta(os.path.dirname(ruta) or ".")
    cal = resultado.calibracion
    tiempos = [p.tiempo for p in resultado.puntos]
    if cal.activa:
        velocidades = [p.velocidad_real for p in resultado.puntos]
        etiqueta_y = f"Velocidad ({cal.unidad_velocidad})"
        vel_prom = resultado.velocidad_promedio_real
    else:
        velocidades = [p.velocidad_px for p in resultado.puntos]
        etiqueta_y = "Velocidad (px/s)"
        vel_prom = resultado.velocidad_promedio_px

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempos, velocidades, color="#1f77b4", linewidth=1.2, label="Velocidad instantánea")
    if vel_prom > 0:
        ax.axhline(
            vel_prom,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label=f"Promedio: {vel_prom:.2f}",
        )
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel(etiqueta_y)
    ax.set_title("Velocidad del cladócero en el tiempo")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def generar_todas(
    resultado: ResultadoAnalisis,
    ancho: int,
    alto: int,
    carpeta: str,
    prefijo: str = "daphnia",
    frame_fondo: Optional[np.ndarray] = None,
) -> dict:
    """Genera todas las salidas en la carpeta indicada.

    Returns:
        dict con las rutas de cada salida generada.
    """
    _asegurar_carpeta(carpeta)
    rutas = {
        "csv": os.path.join(carpeta, f"{prefijo}_datos.csv"),
        "resumen": os.path.join(carpeta, f"{prefijo}_resumen.txt"),
        "larga_exposicion": os.path.join(carpeta, f"{prefijo}_larga_exposicion.png"),
        "mapa_calor": os.path.join(carpeta, f"{prefijo}_mapa_calor.png"),
        "grafica": os.path.join(carpeta, f"{prefijo}_grafica_velocidad.png"),
    }
    exportar_csv(resultado, rutas["csv"])
    exportar_resumen(resultado, rutas["resumen"])
    generar_larga_exposicion(resultado, ancho, alto, rutas["larga_exposicion"], frame_fondo)
    generar_mapa_calor(resultado, ancho, alto, rutas["mapa_calor"])
    generar_grafica_velocidad(resultado, rutas["grafica"])
    return rutas
