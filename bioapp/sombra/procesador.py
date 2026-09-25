"""Procesador de mapa de calor de ocupación (núcleo de ShadowMotion).

Funciones puras de OpenCV/NumPy/matplotlib, sin consola ni tkinter:
reciben una configuración y opcionalmente callbacks de progreso y
cancelación para integrarse con la interfaz.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np
from matplotlib import colormaps

EXTENSIONES_VIDEO = ("avi", "mp4", "mkv", "mov", "wmv")


@dataclass
class ConfigSombra:
    """Parámetros del análisis (valores por defecto del original)."""

    sample_every: int = 30        # Muestrear fondo cada N frames
    max_samples: int = 200        # Máx. frames para estimar el fondo
    threshold: int = 15           # Umbral de detección (0-255)
    min_area: int = 50            # Área mínima del animal (px)
    blur: int = 7                 # Kernel de blur (impar; 0 = desactivado)
    accumulate_mask: bool = False  # True = máscara completa, False = centroide
    gamma: float = 0.6            # Corrección gamma de la visualización
    alpha: float = 0.55           # Opacidad del heatmap sobre el fondo

    def validar(self) -> Optional[str]:
        """Devuelve un mensaje de error si la configuración es inválida."""
        if self.sample_every < 1:
            return "sample_every debe ser >= 1"
        if self.max_samples < 1:
            return "max_samples debe ser >= 1"
        if not (0 <= self.threshold <= 255):
            return "El umbral debe estar entre 0 y 255"
        if self.min_area < 1:
            return "min_area debe ser >= 1"
        return None


def build_background(cap, sample_every: int = 30, max_samples: int = 200):
    """Estima el fondo como la mediana de frames muestreados.

    Funciona bien si el animal no ocupa SIEMPRE el mismo lugar.
    """
    frames = []

    i = 0
    taken = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            frames.append(frame)
            taken += 1
            if taken >= max_samples:
                break
        i += 1

    if len(frames) == 0:
        raise RuntimeError("No se pudieron leer frames para construir el fondo.")

    stack = np.stack(frames, axis=0).astype(np.uint8)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg


def largest_contour_mask(bin_img, min_area: int = 200):
    """Devuelve la máscara y el centroide del contorno más grande."""
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, None

    mask = np.zeros_like(bin_img)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    M = cv2.moments(c)
    if M["m00"] == 0:
        return mask, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return mask, (cx, cy)


def listar_videos_en_carpeta(folder: str) -> List[str]:
    """Lista los videos (varias extensiones) dentro de una carpeta."""
    paths: List[str] = []
    for e in EXTENSIONES_VIDEO:
        for pat in (f"*.{e}", f"*.{e.upper()}"):
            paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(paths))

def procesar_video(
    video_path: str,
    cfg: ConfigSombra,
    output_dir: str,
    on_frame=None,
    on_progreso=None,
    cancel_flag=None,
):
    """Procesa un video y guarda el mapa de calor (NPZ + PNG).

    Args:
        video_path: ruta del video.
        cfg: configuración del análisis.
        output_dir: carpeta donde se guardan las salidas.
        on_frame: callback(overlay_rgb) opcional para previsualizar.
        on_progreso: callback(actual, total) opcional.
        cancel_flag: callable() -> bool para cancelar entre frames.

    Returns:
        dict con fps, rutas generadas y preview del overlay (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    # Fondo por video
    bg = build_background(
        cap, sample_every=cfg.sample_every, max_samples=cfg.max_samples
    )
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

    cap.release()
    cap = cv2.VideoCapture(video_path)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    heat = np.zeros((h, w), dtype=np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    indice = 0
    while True:
        if cancel_flag is not None and cancel_flag():
            cap.release()
            return None

        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cfg.blur and cfg.blur >= 3:
            k = cfg.blur if cfg.blur % 2 == 1 else cfg.blur + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        diff = cv2.absdiff(gray, bg_gray)
        _, bin_img = cv2.threshold(diff, cfg.threshold, 255, cv2.THRESH_BINARY)

        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        mask, centroid = largest_contour_mask(bin_img, min_area=cfg.min_area)

        if cfg.accumulate_mask:
            if mask is not None:
                heat += mask.astype(np.float32) / 255.0
        else:
            if centroid is not None:
                cx, cy = centroid
                heat[cy, cx] += 1.0

        indice += 1
        if on_progreso is not None and total > 0:
            on_progreso(indice, total)

    cap.release()

    heat_norm = heat.copy()
    if heat_norm.max() > 0:
        heat_norm /= heat_norm.max()

    gamma = cfg.gamma
    heat_vis = np.power(heat_norm, gamma)

    cmap = colormaps["inferno"]
    heat_rgb = (cmap(heat_vis)[..., :3] * 255).astype(np.uint8)

    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    alpha = cfg.alpha
    overlay = (
        bg_rgb.astype(np.float32) * (1 - alpha)
        + heat_rgb.astype(np.float32) * alpha
    ).astype(np.uint8)

    base = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # heat en "frames" y en "segundos"
    heat_frames = heat.astype(np.float32)
    heat_seconds = heat_frames / fps if fps and fps > 0 else heat_frames

    npz_path = os.path.join(output_dir, f"{base}_heatmap.npz")
    np.savez_compressed(
        npz_path,
        heat_frames=heat_frames,
        heat_seconds=heat_seconds,
        fps=np.float32(fps),
        height=np.int32(heat.shape[0]),
        width=np.int32(heat.shape[1]),
        accumulate_mask=np.bool_(cfg.accumulate_mask),
    )

    heat_png = os.path.join(output_dir, f"{base}_heatmap.png")
    from matplotlib import pyplot as plt

    plt.imsave(heat_png, heat_vis, cmap="inferno")

    if on_frame is not None:
        on_frame(overlay)

    return {
        "fps": fps,
        "npz": npz_path,
        "heat_png": heat_png,
        "overlay": overlay,
        "base": base,
    }
