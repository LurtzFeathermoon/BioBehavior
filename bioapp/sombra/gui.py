"""Pestaña de mapa de calor de ocupación (PySide6, sin consola ni tkinter).

Flujo: elegir un video o una carpeta con videos → ajustar parámetros →
procesar en un hilo → guardar por video un NPZ numérico y un PNG con el
mapa de calor, mostrando una vista previa del overlay en el visor.
"""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..comunes import mostrar_frame_en
from .procesador import ConfigSombra, listar_videos_en_carpeta, procesar_video


class HiloSombra(QThread):
    """Procesa uno o varios videos en un hilo sin bloquear la GUI."""

    log = Signal(str)                   # una línea de resultado por video
    estado = Signal(str)
    progreso = Signal(int, int)         # frames (actual, total) del video actual
    video_actual = Signal(str, int, int)  # nombre, índice, total de videos
    preview = Signal(object)            # overlay RGB listo para el visor
    terminado = Signal()
    error = Signal(str)

    def __init__(
        self,
        videos: List[str],
        cfg: ConfigSombra,
        carpeta_salida: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.videos = videos
        self.cfg = cfg
        self.carpeta_salida = carpeta_salida
        self._cancelar = False

    def cancelar(self) -> None:
        self._cancelar = True

    def run(self) -> None:
        total = len(self.videos)
        try:
            for i, video in enumerate(self.videos, start=1):
                if self._cancelar:
                    self.estado.emit("Análisis cancelado.")
                    break
                nombre = os.path.basename(video)
                self.video_actual.emit(nombre, i, total)
                self.estado.emit(f"Procesando ({i}/{total}): {nombre}")

                try:
                    res = procesar_video(
                        video,
                        self.cfg,
                        self.carpeta_salida,
                        on_frame=self.preview.emit,
                        on_progreso=self.progreso.emit,
                        cancel_flag=lambda: self._cancelar,
                    )
                except Exception as e:  # noqa: BLE001
                    self.log.emit(f"ERROR: {nombre} -> {e}")
                    continue

                if res is None:
                    self.estado.emit("Análisis cancelado.")
                    break

                self.log.emit(
                    f"[{i}/{total}] OK: {nombre}  (FPS: {res['fps']:.3f})"
                )
                self.log.emit(f"   · {res['npz']}")
                self.log.emit(f"   · {res['heat_png']}")
        except Exception as e:  # noqa: BLE001
            self.error.emit(str(e))
            return
        self.terminado.emit()

class PanelSombra(QWidget):
    """Pestaña de mapa de calor, con la misma gramática que las demás."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.entrada: Optional[str] = None
        self.es_carpeta = False
        self.carpeta_salida: Optional[str] = None
        self.hilo: Optional[HiloSombra] = None
        self._construir_ui()

    # ------------------------------------------------------------------ UI
    def _construir_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        panel = QWidget()
        panel.setFixedWidth(380)
        pl = QVBoxLayout(panel)
        pl.setSpacing(10)

        # 1. Entrada
        gb_entrada = QGroupBox("1. Entrada")
        el = QVBoxLayout(gb_entrada)
        self.rb_video = QRadioButton("Un solo video")
        self.rb_carpeta = QRadioButton("Carpeta completa (lote)")
        self.rb_video.setChecked(True)
        grupo = QButtonGroup(self)
        grupo.addButton(self.rb_video)
        grupo.addButton(self.rb_carpeta)
        self.btn_entrada = QPushButton("Seleccionar video...")
        self.btn_entrada.clicked.connect(self.seleccionar_entrada)
        self.lbl_entrada = QLabel("Ningún video seleccionado")
        self.lbl_entrada.setWordWrap(True)
        self.lbl_entrada.setObjectName("info")
        el.addWidget(self.rb_video)
        el.addWidget(self.rb_carpeta)
        el.addWidget(self.btn_entrada)
        el.addWidget(self.lbl_entrada)
        self.rb_carpeta.toggled.connect(self._actualizar_modo)
        pl.addWidget(gb_entrada)

        # 2. Parámetros
        gb_param = QGroupBox("2. Parámetros")
        fl = QFormLayout(gb_param)
        self.txt_sample_every = QLineEdit("30")
        self.txt_max_samples = QLineEdit("200")
        self.txt_threshold = QLineEdit("15")
        self.txt_min_area = QLineEdit("50")
        self.txt_blur = QLineEdit("7")
        for w in (
            self.txt_sample_every,
            self.txt_max_samples,
            self.txt_threshold,
            self.txt_min_area,
            self.txt_blur,
        ):
            w.setValidator(QIntValidator(0, 10_000_000, self))
        fl.addRow("Muestrear fondo cada:", self.txt_sample_every)
        fl.addRow("Máx. frames para fondo:", self.txt_max_samples)
        fl.addRow("Umbral de detección:", self.txt_threshold)
        fl.addRow("Área mínima (px):", self.txt_min_area)
        fl.addRow("Kernel de blur (0=no):", self.txt_blur)
        self.chk_acumular = QCheckBox(
            "Acumular máscara completa (si no → centroide)"
        )
        fl.addRow(self.chk_acumular)
        ayuda = QLabel(
            "Fondo = mediana de frames muestreados. Cada video genera "
            "<nombre>_heatmap.npz (numérico) y <nombre>_heatmap.png."
        )
        ayuda.setWordWrap(True)
        ayuda.setObjectName("ayuda")
        fl.addRow(ayuda)
        pl.addWidget(gb_param)

        # 3. Carpeta de salida
        gb_out = QGroupBox("3. Carpeta de salida")
        ol = QVBoxLayout(gb_out)
        self.btn_out = QPushButton("Seleccionar carpeta...")
        self.btn_out.clicked.connect(self.seleccionar_salida)
        self.lbl_out = QLabel("Ninguna carpeta seleccionada")
        self.lbl_out.setWordWrap(True)
        self.lbl_out.setObjectName("info")
        ol.addWidget(self.btn_out)
        ol.addWidget(self.lbl_out)
        pl.addWidget(gb_out)

        self.btn_run = QPushButton("▶ Procesar mapa de calor")
        self.btn_run.setObjectName("primario")
        self.btn_run.clicked.connect(self.iniciar)
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancelar)
        pl.addWidget(self.btn_run)
        pl.addWidget(self.btn_cancel)
        pl.addStretch(1)
        layout.addWidget(panel)

        # ---- Panel derecho: visor, progreso y bitácora ----
        derecho = QVBoxLayout()
        self.visor = QLabel(
            "La vista previa del mapa de calor aparecerá aquí al procesar."
        )
        self.visor.setAlignment(Qt.AlignCenter)
        self.visor.setObjectName("visor")
        self.visor.setMinimumSize(560, 260)
        derecho.addWidget(self.visor, stretch=2)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        derecho.addWidget(self.progress)

        self.lbl_estado = QLabel("Listo. Elija la entrada y la carpeta de salida.")
        self.lbl_estado.setObjectName("estado")
        self.lbl_estado.setWordWrap(True)
        derecho.addWidget(self.lbl_estado)

        self.resultados = QPlainTextEdit()
        self.resultados.setObjectName("resultados")
        self.resultados.setReadOnly(True)
        self.resultados.setPlaceholderText("Bitácora del procesamiento...")
        self.resultados.setMaximumHeight(220)
        derecho.addWidget(self.resultados, stretch=1)
        layout.addLayout(derecho, stretch=1)

    def _actualizar_modo(self) -> None:
        self.es_carpeta = self.rb_carpeta.isChecked()
        self.btn_entrada.setText(
            "Seleccionar carpeta con videos..." if self.es_carpeta
            else "Seleccionar video..."
        )
        self.entrada = None
        self.lbl_entrada.setText(
            "Carpeta con videos (.mp4, .avi, .mkv, .mov)..." if self.es_carpeta
            else "Ningún video seleccionado"
        )

    # -------------------------------------------------------------- acciones
    def seleccionar_entrada(self) -> None:
        if self.es_carpeta:
            carpeta = QFileDialog.getExistingDirectory(
                self, "Seleccionar carpeta con videos"
            )
            if not carpeta:
                return
            videos = listar_videos_en_carpeta(carpeta)
            if not videos:
                QMessageBox.warning(
                    self,
                    "Sin videos",
                    "La carpeta no contiene videos "
                    "(.mp4, .avi, .mkv, .mov, .wmv).",
                )
                return
            self.entrada = carpeta
            self.lbl_entrada.setText(
                f"<b>{os.path.basename(carpeta)}</b><br>"
                f"{len(videos)} video(s) encontrados"
            )
        else:
            video, _ = QFileDialog.getOpenFileName(
                self,
                "Seleccionar video",
                "",
                "Videos (*.mp4 *.avi *.mkv *.mov *.wmv)",
            )
            if not video:
                return
            self.entrada = video
            self.lbl_entrada.setText(
                f"<b>{os.path.basename(video)}</b>"
            )

    def seleccionar_salida(self) -> None:
        carpeta = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta donde guardar los resultados"
        )
        if not carpeta:
            return
        self.carpeta_salida = carpeta
        self.lbl_out.setText(f"<b>{carpeta}</b>")

    def _leer_cfg(self):
        def valor(txt):
            return int(txt.text().strip() or "0")

        cfg = ConfigSombra(
            sample_every=valor(self.txt_sample_every),
            max_samples=valor(self.txt_max_samples),
            threshold=valor(self.txt_threshold),
            min_area=valor(self.txt_min_area),
            blur=valor(self.txt_blur),
            accumulate_mask=self.chk_acumular.isChecked(),
        )
        error = cfg.validar()
        if error:
            raise ValueError(error)
        return cfg

    def iniciar(self) -> None:
        if self.hilo is not None and self.hilo.isRunning():
            return

        try:
            cfg = self._leer_cfg()
        except ValueError as e:
            QMessageBox.warning(self, "Parámetros inválidos", str(e))
            return

        if not self.entrada:
            QMessageBox.warning(
                self, "Entrada requerida", "Seleccione un video o una carpeta."
            )
            return
        if not self.carpeta_salida:
            QMessageBox.warning(
                self, "Salida requerida", "Seleccione la carpeta de salida."
            )
            return

        videos: List[str] = (
            listar_videos_en_carpeta(self.entrada)
            if self.es_carpeta
            else [self.entrada]
        )
        if not videos:
            QMessageBox.warning(
                self, "Sin videos", "No hay videos que procesar."
            )
            return

        self.resultados.clear()
        self.progress.setValue(0)
        self._bloquear_controles(True)

        self.hilo = HiloSombra(videos, cfg, self.carpeta_salida, self)
        self.hilo.log.connect(self._agregar_log)
        self.hilo.estado.connect(self.lbl_estado.setText)
        self.hilo.progreso.connect(self.progress.setValue)
        self.hilo.preview.connect(self._mostrar_preview)
        self.hilo.terminado.connect(self._al_terminar)
        self.hilo.error.connect(self._al_error)
        self.hilo.start()

    def cancelar(self) -> None:
        if self.hilo is not None and self.hilo.isRunning():
            self.hilo.cancelar()
            self.lbl_estado.setText("Cancelando...")

    # -------------------------------------------------------------- callbacks
    def _agregar_log(self, linea: str) -> None:
        self.resultados.appendPlainText(linea)

    def _mostrar_preview(self, overlay) -> None:
        mostrar_frame_en(self.visor, overlay)

    def _al_terminar(self) -> None:
        self._bloquear_controles(False)
        self.progress.setValue(0)
        self.lbl_estado.setText("Procesamiento finalizado. Revisar bitácora.")

    def _al_error(self, msg: str) -> None:
        self._bloquear_controles(False)
        self.lbl_estado.setText("Error durante el procesamiento.")
        QMessageBox.critical(self, "Error", msg)

    def _bloquear_controles(self, procesando: bool) -> None:
        self.btn_run.setEnabled(not procesando)
        self.btn_entrada.setEnabled(not procesando)
        self.btn_out.setEnabled(not procesando)
        self.btn_cancel.setEnabled(procesando)
        self.rb_video.setEnabled(not procesando)
        self.rb_carpeta.setEnabled(not procesando)

    def cerrar(self) -> None:
        """Cancela el hilo activo (llamado al cerrar la aplicación)."""
        if self.hilo is not None and self.hilo.isRunning():
            self.hilo.cancelar()
            self.hilo.wait(2000)
