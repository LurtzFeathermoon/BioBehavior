"""Gráfica de la señal cardíaca dibujada con QPainter (sin matplotlib)."""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


class SignalPlot(QWidget):
    def __init__(self):
        super().__init__()

        # ==== CONFIGURACIÓN CRÍTICA ====
        self.setMinimumHeight(220)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(True)

        self.signal = None
        self.peaks = []

        # Cuadrícula
        self.grid_x = 10   # divisiones en tiempo
        self.grid_y = 5    # divisiones en amplitud

    def set_data(self, signal, peaks):
        self.signal = signal
        self.peaks = peaks
        self.update()  # FORZAR REPINTADO

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_plot(painter, self.rect())

        rect = self.rect()
        w = rect.width()
        h = rect.height()

        # ==== FONDO ====
        painter.fillRect(rect, Qt.white)

        if self.signal is None or len(self.signal) < 2:
            painter.end()
            return

        # ==== CUADRÍCULA ====
        grid_pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(grid_pen)

        for i in range(1, self.grid_x):
            x = int(i * w / self.grid_x)
            painter.drawLine(x, 0, x, h)

        for j in range(1, self.grid_y):
            y = int(j * h / self.grid_y)
            painter.drawLine(0, y, w, y)

        # ==== EJES ====
        axis_pen = QPen(Qt.black, 2)
        painter.setPen(axis_pen)
        painter.drawLine(0, h - 1, w, h - 1)  # eje X
        painter.drawLine(0, 0, 0, h)          # eje Y

        # ==== ETIQUETAS ====
        painter.setFont(QFont("Arial", 9))
        painter.drawText(5, 15, "Intensidad")
        painter.drawText(w - 40, h - 5, "Tiempo")

        # ==== SEÑAL ====
        n = len(self.signal)
        signal_pen = QPen(Qt.black, 2)
        painter.setPen(signal_pen)

        for i in range(n - 1):
            x1 = int(i * w / (n - 1))
            y1 = int((1 - self.signal[i]) * h)
            x2 = int((i + 1) * w / (n - 1))
            y2 = int((1 - self.signal[i + 1]) * h)
            painter.drawLine(x1, y1, x2, y2)

        # ==== PICOS ====
        peak_pen = QPen(Qt.red, 3)
        painter.setPen(peak_pen)

        for p in self.peaks:
            x = int(p * w / (n - 1))
            y = int((1 - self.signal[p]) * h)
            painter.drawPoint(x, y)

        painter.end()

    def save_png(self, filename, width=1920, height=1080):
        image = QImage(width, height, QImage.Format_ARGB32)
        image.fill(Qt.white)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_plot(painter,QRect(0,0,width,height))

        painter.end()
        image.save(filename)

    def _draw_plot(self, painter, rect):
        if self.signal is None or len(self.signal) == 0:
            return

        w = rect.width()
        h = rect.height()
        margin = 60

        plot_w = w - 2 * margin
        plot_h = h - 2 * margin

        # Fondo
        painter.fillRect(rect, Qt.white)

        # ==== CUADRÍCULA ====
        grid_pen = QPen(Qt.lightGray, 1, Qt.DashLine)
        painter.setPen(grid_pen)

        for i in range(1, self.grid_x):
            x = int(i * w / self.grid_x)
            painter.drawLine(x, 0, x, h)

        for j in range(1, self.grid_y):
            y = int(j * h / self.grid_y)
            painter.drawLine(0, y, w, y)

        # ==== EJES ====
        axis_pen = QPen(Qt.black, 2)
        painter.setPen(axis_pen)
        painter.drawLine(0, h - 1, w, h - 1)  # eje X
        painter.drawLine(0, 0, 0, h)          # eje Y

        # ==== ETIQUETAS ====
        painter.setFont(QFont("Arial", 9))
        painter.drawText(5, 15, "Intensidad")
        painter.drawText(w - 40, h - 5, "Tiempo")

        # ==== SEÑAL ====
        n = len(self.signal)
        signal_pen = QPen(Qt.black, 2)
        painter.setPen(signal_pen)

        for i in range(n - 1):
            x1 = int(i * w / (n - 1))
            y1 = int((1 - self.signal[i]) * h)
            x2 = int((i + 1) * w / (n - 1))
            y2 = int((1 - self.signal[i + 1]) * h)
            painter.drawLine(x1, y1, x2, y2)

        # ==== PICOS ====
        peak_pen = QPen(Qt.red, 6)
        painter.setPen(peak_pen)

        for p in self.peaks:
            x = int(p * w / (n - 1))
            y = int((1 - self.signal[p]) * h)
            painter.drawPoint(x, y)
