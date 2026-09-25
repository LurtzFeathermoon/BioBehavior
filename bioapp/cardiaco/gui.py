"""Pestaña de análisis de pulso cardíaco de daphnias (PySide6).

Flujo: cargar video → dibujar la región de interés (tórax) sobre el
primer frame → analizar (en un hilo) → ver BPM y señal → exportar
CSV + PNG. Conserva el procesamiento y el formato de salida originales
de DaphniaHeartRate.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional

import cv2
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .grafica import SignalPlot
from .procesador import process_video
from .roi import ROISelector


class VideoWorker(QObject):
    """Analiza el video en un hilo separado, emitiendo progreso y frames."""

    finished = Signal(object)
    progress = Signal(int)
    error = Signal(str)
    frame_ready = Signal(object, object)

    def __init__(self, video_path, roi, accelerated, seguir_automatico=False):
        super().__init__()
        self.video_path = video_path
        self.roi = roi
        self.accelerated = accelerated
        self.seguir_automatico = seguir_automatico
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = process_video(
                self.video_path,
                self.roi,
                self.accelerated,
                progress_cb=self.progress.emit,
                frame_cb=self.frame_ready.emit,
                cancel_flag=lambda: self._cancel,
                seguir_automatico=self.seguir_automatico,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
class PanelCardiaco(QWidget):
    """Pestaña de pulso cardíaco, con la misma gramática que el rastreo."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.video_path: Optional[str] = None
        self.roi: Optional[tuple] = None
        self.thread: Optional[QThread] = None
        self.worker: Optional[VideoWorker] = None
        self.last_result = None
        self.last_metadata = {}
        self._construir_ui()

    # ------------------------------------------------------------------ UI
    def _construir_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        panel = QWidget()
        panel.setFixedWidth(380)
        pl = QVBoxLayout(panel)
        pl.setSpacing(10)

        gb_video = QGroupBox("1. Video")
        vl = QVBoxLayout(gb_video)
        self.btn_select = QPushButton("Cargar video (.mp4, .avi, .mov, .mkv)")
        self.btn_select.clicked.connect(self.select_video)
        self.lbl_video = QLabel("Ningún video cargado")
        self.lbl_video.setWordWrap(True)
        self.lbl_video.setObjectName("info")
        vl.addWidget(self.btn_select)
        vl.addWidget(self.lbl_video)
        pl.addWidget(gb_video)

        gb_roi = QGroupBox("2. Región de interés (tórax)")
        rl = QVBoxLayout(gb_roi)
        ayuda = QLabel(
            "Sobre el video de la derecha, arrastre el ratón para enmarcar "
            "la región donde late el corazón (el tórax de la daphnia)."
        )
        ayuda.setWordWrap(True)
        ayuda.setObjectName("ayuda")
        self.chk_realtime = QCheckBox("Análisis en tiempo real")
        self.chk_realtime.setChecked(False)
        ayuda2 = QLabel(
            "Desactivado = análisis acelerado (recomendado). "
            "Activado = dibuja el ROI sobre cada frame (más lento)."
        )
        ayuda2.setWordWrap(True)
        ayuda2.setObjectName("ayuda")
        rl.addWidget(ayuda)
        rl.addWidget(self.chk_realtime)
        rl.addWidget(ayuda2)

        self.chk_seguimiento = QCheckBox("Seguimiento automático (forma y giro)")
        self.chk_seguimiento.setChecked(False)
        ayuda3 = QLabel(
            "Detecta la silueta del organismo y mueve/rota la ROI con él "
            "para no perder el corazón si se desplaza o gira sobre sí "
            "mismo. Funciona con cualquier organismo, pero conserva el "
            "tamaño del área elegida (no vuelve a buscar el corazón "
            "dentro del cuerpo)."
        )
        ayuda3.setWordWrap(True)
        ayuda3.setObjectName("ayuda")
        rl.addWidget(self.chk_seguimiento)
        rl.addWidget(ayuda3)
        pl.addWidget(gb_roi)

        gb_out = QGroupBox("3. Resultados y exportación")
        ol = QVBoxLayout(gb_out)
        self.result_label = QLabel("BPM: --")
        self.result_label.setObjectName("estado")
        self.output_name = QLineEdit()
        self.output_name.setPlaceholderText("Nombre de la muestra (CSV/PNG)")
        ol.addWidget(self.result_label)
        ol.addWidget(self.output_name)
        self.btn_export = QPushButton("Exportar resultados (CSV + PNG)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results)
        ol.addWidget(self.btn_export)
        pl.addWidget(gb_out)

        self.btn_run = QPushButton("▶ Analizar pulso")
        self.btn_run.setObjectName("primario")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        pl.addWidget(self.btn_run)
        pl.addWidget(self.btn_cancel)
        pl.addStretch(1)
        layout.addWidget(panel)

        # ---- Panel derecho: visor, señal y estado ----
        derecho = QVBoxLayout()
        self.video_label = ROISelector()
        self.video_label.setObjectName("visor")
        self.video_label.setMinimumSize(560, 300)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        derecho.addWidget(self.video_label, stretch=3)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        derecho.addWidget(self.progress)

        self.lbl_estado = QLabel("Listo. Cargue un video y dibuje la región de interés.")
        self.lbl_estado.setObjectName("estado")
        self.lbl_estado.setWordWrap(True)
        derecho.addWidget(self.lbl_estado)

        self.plot = SignalPlot()
        derecho.addWidget(self.plot, stretch=2)
        layout.addLayout(derecho, stretch=1)

    # -------------------------------------------------------------- acciones
    def select_video(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not file:
            return

        cap = cv2.VideoCapture(file)
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        if not ret:
            QMessageBox.critical(self, "Error", "No se pudo leer el video")
            return

        self.video_path = file
        self.lbl_video.setText(
            f"<b>{os.path.basename(file)}</b><br>"
            f"{frame.shape[1]}x{frame.shape[0]} px · {fps:.1f} FPS · "
            f"{total} frames"
        )
        self.video_label.setObjectName("visor")
        self.video_label.show_frame(frame)
        self.video_label.image_size = (frame.shape[1], frame.shape[0])
        self.result_label.setText("BPM: --")
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_estado.setText("Dibuje la región de interés sobre el video.")

    def show_realtime_frame(self, frame, roi) -> None:
        if len(roi) == 5:
            cx, cy, w, h, angulo = roi
            caja = cv2.boxPoints(((cx, cy), (w, h), angulo))
            caja = caja.astype(int)
            cv2.polylines(frame, [caja], True, (0, 255, 0), 2)
        else:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.video_label.show_frame(frame)

    def run_analysis(self) -> None:
        self.roi = self.video_label.get_roi_cv()
        if not self.video_path or not self.roi:
            QMessageBox.warning(
                self,
                "Faltan datos",
                "Seleccione un video y dibuje la región de interés (ROI).",
            )
            return

        if self.thread is not None and self.thread.isRunning():
            return

        accelerated = not self.chk_realtime.isChecked()
        self._bloquear_controles(True)
        self.lbl_estado.setText("Analizando pulso...")
        self.progress.setValue(0)

        seguir_automatico = self.chk_seguimiento.isChecked()
        self.thread = QThread()
        self.worker = VideoWorker(self.video_path, self.roi, accelerated, seguir_automatico)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.show_realtime_frame)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.thread.start()

    # -------------------------------------------------------------- callbacks
    def export_results(self) -> None:
        if not hasattr(self, "last_result") or self.last_result is None:
            return

        name = self.output_name.text().strip()
        if not name:
            QMessageBox.warning(
                self,
                "Nombre requerido",
                "Ingrese un nombre para los archivos de salida.",
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de destino"
        )
        if not folder:
            return

        csv_path = os.path.join(folder, f"{name}.csv")
        png_path = os.path.join(folder, f"{name}.png")

        bpm, signal, peaks, fps = self.last_result
        self.save_csv(csv_path, bpm, fps, self.last_metadata)
        self.plot.save_png(png_path)

        QMessageBox.information(
            self,
            "Exportación completa",
            f"Archivos guardados:\n{name}.csv\n{name}.png",
        )

    def save_csv(self, path, bpm, fps, metadata=None) -> None:
        metadata = metadata or {}

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "video",
                    "bpm",
                    "fps",
                    "roi_x",
                    "roi_y",
                    "roi_w",
                    "roi_h",
                    "signal_source",
                    "peak_count",
                    "median_interval_frames",
                    "median_interval_sec",
                    "bpm_from_median_interval",
                    "interval_cv",
                    "median_prominence",
                    "acf_lag_frames",
                    "acf_strength",
                    "score",
                    "bpm_min",
                    "bpm_max",
                    "hit_bpm_ceiling",
                ]
            )
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    self.video_path,
                    f"{bpm:.2f}",
                    fps,
                    *self.roi,
                    metadata.get("signal_source", ""),
                    metadata.get("peak_count", ""),
                    metadata.get("median_interval_frames", ""),
                    metadata.get("median_interval_sec", ""),
                    metadata.get("bpm_from_median_interval", ""),
                    metadata.get("interval_cv", ""),
                    metadata.get("median_prominence", ""),
                    metadata.get("acf_lag_frames", ""),
                    metadata.get("acf_strength", ""),
                    metadata.get("score", ""),
                    metadata.get("bpm_min", ""),
                    metadata.get("bpm_max", ""),
                    metadata.get("hit_bpm_ceiling", ""),
                ]
            )

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.lbl_estado.setText("Cancelando...")

    def _bloquear_controles(self, procesando: bool) -> None:
        self.btn_run.setEnabled(not procesando)
        self.btn_select.setEnabled(not procesando)
        self.btn_cancel.setEnabled(procesando)
        self.chk_realtime.setEnabled(not procesando)
        self.chk_seguimiento.setEnabled(not procesando)

    def on_finished(self, result) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self._bloquear_controles(False)

        if result is None:
            self.lbl_estado.setText("Análisis cancelado.")
            return

        metadata = {}
        if len(result) >= 5:
            bpm, signal, peaks, fps, metadata = result
        else:
            bpm, signal, peaks, fps = result

        source = metadata.get("signal_source")
        ceiling = metadata.get("hit_bpm_ceiling")
        parts = [f"BPM estimado: {bpm:.1f}"]
        if source:
            parts.append(f"fuente: {source}")
        if ceiling:
            parts.append("cerca del límite superior")
        self.result_label.setText(" | ".join(parts))

        self.plot.set_data(signal, peaks)
        self.last_result = (bpm, signal, peaks, fps)
        self.last_metadata = metadata
        self.btn_export.setEnabled(True)
        self.lbl_estado.setText("Análisis completado. Puede exportar los resultados.")

    def on_error(self, msg) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self._bloquear_controles(False)
        self.lbl_estado.setText("Error durante el análisis.")
        QMessageBox.critical(self, "Error", msg)

    def cerrar(self) -> None:
        """Cancela el hilo activo (llamado al cerrar la aplicación)."""
        if self.worker is not None:
            self.worker.cancel()
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
