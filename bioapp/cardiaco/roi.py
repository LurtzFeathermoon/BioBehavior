"""Selector de región de interés (ROI) sobre el primer frame del video."""

from __future__ import annotations

import cv2
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QRubberBand


class ROISelector(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)

        self.setScaledContents(False)
        self._current_frame = None

        self.display_rect = QRect()
        self.origin = QPoint()
        self.rubber = QRubberBand(QRubberBand.Rectangle, self)
        self.roi_rect = None
        self.image_size = None

    def set_image(self, pixmap, image_size):
        self.setPixmap(pixmap)
        self.update_display_rect()
        self.image_size = image_size
        self.roi_rect = None
        self.rubber.hide()

    def mousePressEvent(self, event):
        if self.pixmap() is None:
            return
        if not self.display_rect.contains(event.pos()):
            return
        self.origin = event.pos()
        self.rubber.setGeometry(QRect(self.origin, QSize()))
        self.rubber.show()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        pos.setX(max(self.display_rect.left(), min(pos.x(), self.display_rect.right())))
        pos.setY(max(self.display_rect.top(), min(pos.y(), self.display_rect.bottom())))

        self.rubber.setGeometry(QRect(self.origin, pos).normalized())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display_rect()

    def update_display_rect(self):
        if not self.pixmap():
            return

        pm = self.pixmap()
        pm_size = pm.size()
        label_size = self.size()

        scaled = pm_size.scaled(label_size, Qt.KeepAspectRatio)

        x = (label_size.width() - scaled.width()) // 2
        y = (label_size.height() - scaled.height()) // 2

        self.display_rect = QRect(
            x, y,
            scaled.width(),
            scaled.height()
        )

    def mouseReleaseEvent(self, event):
        self.roi_rect = self.rubber.geometry()
        self.rubber.hide()

    def get_roi_cv(self):
        if not self.roi_rect or self.display_rect.isNull():
            return None

        img_w, img_h = self.image_size
        disp = self.display_rect

        scale_x = img_w / disp.width()
        scale_y = img_h / disp.height()

        x = int((self.roi_rect.x() - disp.x()) * scale_x)
        y = int((self.roi_rect.y() - disp.y()) * scale_y)
        w = int(self.roi_rect.width() * scale_x)
        h = int(self.roi_rect.height() * scale_y)

        return x, y, w, h

    def show_frame(self, frame_bgr):
        self._current_frame = frame_bgr

        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)

        scaled = pix.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled)

        # calcular display_rect REAL
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self.display_rect = QRect(
            x, y,
            scaled.width(),
            scaled.height()
        )

        self.update()

# ====
# SIGNAL PLOT
# ====
