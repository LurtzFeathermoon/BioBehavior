"""Módulo de procesamiento de video con OpenCV.

Abre el video, recorre los frames, ejecuta el detector y construye la
trayectoria. Genera frames con la trayectoria superpuesta y reporta el
progreso mediante callbacks (para integrarse con la GUI).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .tracker import DetectorCladocero, DeteccionFrame, ParametrosDeteccion, Trayectoria

FORMATOS_SOPORTADOS = (".mp4", ".avi", ".mov")


@dataclass
class InfoVideo:
    """Metadatos del video."""

    ruta: str
    fps: float
    total_frames: int
    ancho: int
    alto: int
    duracion: float


class ErrorVideo(Exception):
    """Error al abrir o procesar un video."""


def validar_formato(ruta: str) -> bool:
    return os.path.splitext(ruta)[1].lower() in FORMATOS_SOPORTADOS


def leer_info_video(ruta: str) -> InfoVideo:
    """Lee los metadatos del video sin procesarlo."""
    if not os.path.isfile(ruta):
        raise ErrorVideo(f"No se encontró el archivo de video:\n{ruta}")
    if not validar_formato(ruta):
        raise ErrorVideo(
            "Formato no soportado. Use uno de: "
            + ", ".join(FORMATOS_SOPORTADOS)
        )
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        cap.release()
        raise ErrorVideo(
            "No se pudo abrir el video. Verifique que el archivo no esté "
            "dañado y que el códec sea compatible."
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    duracion = total / fps if fps else 0.0
    return InfoVideo(ruta, fps, total, ancho, alto, duracion)


class ProcesadorVideo:
    """Procesa un video completo y construye la trayectoria del cladócero."""

    def __init__(
        self,
        detector: Optional[DetectorCladocero] = None,
    ) -> None:
        self.detector = detector or DetectorCladocero()
        self._cancelar = False

    def cancelar(self) -> None:
        self._cancelar = True

    def reset_cancelar(self) -> None:
        self._cancelar = False

    @staticmethod
    def dibujar_overlay(
        frame: np.ndarray,
        trayectoria: Trayectoria,
        deteccion_actual: Optional[DeteccionFrame],
        max_puntos: int = 400,
    ) -> np.ndarray:
        """Dibuja la trayectoria acumulada y la posición actual sobre el frame."""
        salida = frame.copy()
        puntos = trayectoria.coordenadas()
        if len(puntos) >= 2:
            recientes = puntos[-max_puntos:]
            pts = np.array(recientes, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(salida, [pts], False, (0, 200, 255), 2, cv2.LINE_AA)
        if deteccion_actual is not None and deteccion_actual.detectado:
            cx, cy = int(deteccion_actual.x), int(deteccion_actual.y)
            cv2.circle(salida, (cx, cy), 8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(
                salida, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 14, 1
            )
        return salida

    def procesar(
        self,
        ruta: str,
        on_frame: Optional[Callable[[int, int, np.ndarray, DeteccionFrame], None]] = None,
        on_progreso: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Trayectoria, InfoVideo]:
        """Procesa el video completo.

        Args:
            ruta: ruta al archivo de video.
            on_frame: callback(indice, total, frame_overlay, deteccion) por frame.
            on_progreso: callback(indice, total) para barra de progreso.

        Returns:
            (Trayectoria, InfoVideo)
        """
        self.reset_cancelar()
        info = leer_info_video(ruta)
        cap = cv2.VideoCapture(ruta)
        if not cap.isOpened():
            cap.release()
            raise ErrorVideo("No se pudo abrir el video para procesamiento.")

        trayectoria = Trayectoria()
        indice = 0
        total = info.total_frames

        try:
            while True:
                if self._cancelar:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                tiempo = indice / info.fps
                deteccion, _mascara = self.detector.detectar(frame, indice, tiempo)
                trayectoria.agregar(deteccion)

                if on_frame is not None:
                    overlay = self.dibujar_overlay(frame, trayectoria, deteccion)
                    on_frame(indice, total, overlay, deteccion)
                if on_progreso is not None:
                    on_progreso(indice + 1, total)

                indice += 1
        finally:
            cap.release()

        trayectoria.rellenar_huecos()
        return trayectoria, info
