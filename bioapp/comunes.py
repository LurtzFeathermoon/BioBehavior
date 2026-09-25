"""Utilidades compartidas por las pestañas de BioBehaviour."""

from __future__ import annotations

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


def frame_bgr_a_pixmap(frame_bgr) -> QPixmap:
    """Convierte un frame BGR (OpenCV) en un QPixmap RGB."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def mostrar_frame_en(label: QLabel, frame_bgr) -> None:
    """Muestra un frame en un QLabel escalado con proporción y centrado."""
    pix = frame_bgr_a_pixmap(frame_bgr)
    if label.width() > 1 and label.height() > 1:
        pix = pix.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    label.setPixmap(pix)


class PaginaBase(QWidget):
    """Widget base de una pestaña de análisis.

    Cada página gestiona su propio hilo de trabajo y expone ``cerrar()``
    para cancelarlo limpiamente al cerrar la aplicación.
    """

    def __init__(self, titulo: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pagina")
        self.titulo = titulo

    # Sobrescrito por cada pestaña si tiene hilo activo.
    def cerrar(self) -> None:
        pass


def crear_visor(texto_inicial: str) -> QLabel:
    """Crea el visor estándar (marco oscuro) del área derecha."""
    visor = QLabel(texto_inicial)
    visor.setAlignment(Qt.AlignCenter)
    visor.setObjectName("visor")
    visor.setMinimumSize(560, 320)
    visor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return visor
