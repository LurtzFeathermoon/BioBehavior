"""Pestaña de rastreo de movimiento (PySide6) de BioBehaviour.

Panel insertable en la ventana principal con controles para cargar video,
configurar el umbral de movimiento, calibración opcional, visualización
del video con la trayectoria superpuesta, barra de progreso y panel de
resultados. Conserva el flujo y las salidas de DaphniaTracker.
"""

from __future__ import annotations

import os
import re
import traceback
from typing import Optional

import cv2
import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from .analysis import AnalizadorMovimiento, Calibracion, ResultadoAnalisis
from .outputs import generar_todas
from .tracker import DetectorCladocero, DeteccionFrame, ParametrosDeteccion, Trayectoria
from .video_processor import (
    ErrorVideo,
    InfoVideo,
    ProcesadorVideo,
    leer_info_video,
    validar_formato,
)


class HiloProcesamiento(QtCore.QThread):
    """Hilo que procesa el video y ejecuta el análisis sin bloquear la GUI."""

    frame_listo = QtCore.Signal(int, int, np.ndarray, object)
    progreso = QtCore.Signal(int, int)
    estado = QtCore.Signal(str)
    terminado = QtCore.Signal(object, object, object)  # resultado, info, ultimo_frame
    error = QtCore.Signal(str)

    def __init__(
        self,
        ruta_video: str,
        umbral: float,
        calibracion: Calibracion,
        parametros: ParametrosDeteccion,
        carpeta_salida: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.ruta_video = ruta_video
        self.umbral = umbral
        self.calibracion = calibracion
        self.parametros = parametros
        self.carpeta_salida = carpeta_salida
        self._procesador = ProcesadorVideo(DetectorCladocero(parametros))
        self._ultimo_frame: Optional[np.ndarray] = None

    def cancelar(self) -> None:
        self._procesador.cancelar()

    def run(self) -> None:  # noqa: D401
        try:
            self.estado.emit("Procesando video y rastreando cladócero...")

            def on_frame(indice, total, overlay, deteccion):
                self._ultimo_frame = overlay
                self.frame_listo.emit(indice, total, overlay, deteccion)

            def on_progreso(actual, total):
                self.progreso.emit(actual, total)

            trayectoria, info = self._procesador.procesar(
                self.ruta_video, on_frame=on_frame, on_progreso=on_progreso
            )

            if len(trayectoria) == 0:
                self.error.emit(
                    "No se pudieron leer frames del video. Verifique el archivo."
                )
                return

            self.estado.emit("Analizando movimiento...")
            analizador = AnalizadorMovimiento(
                umbral_movimiento=self.umbral, calibracion=self.calibracion
            )
            resultado = analizador.analizar(trayectoria, info.fps)

            self.estado.emit("Generando salidas (CSV, imágenes, gráfica)...")
            generar_todas(
                resultado,
                info.ancho,
                info.alto,
                self.carpeta_salida,
                prefijo=os.path.splitext(os.path.basename(self.ruta_video))[0],
                frame_fondo=self._ultimo_frame,
            )

            self.terminado.emit(resultado, info, self._ultimo_frame)
        except ErrorVideo as e:
            self.error.emit(str(e))
        except Exception as e:  # pragma: no cover
            self.error.emit(f"Error inesperado durante el análisis:\n{e}\n\n{traceback.format_exc()}")


class PanelRastreo(QtWidgets.QWidget):
    """Pestaña de rastreo de movimiento de cladóceros."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.ruta_video: Optional[str] = None
        self.info_video: Optional[InfoVideo] = None
        self.carpeta_salida: Optional[str] = None
        self.hilo: Optional[HiloProcesamiento] = None
        self.resultado: Optional[ResultadoAnalisis] = None

        self._construir_ui()

    # ------------------------------------------------------------------ UI
    def _construir_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # ---- Panel izquierdo: controles ----
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(380)
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setSpacing(10)

        # Grupo: Video
        gb_video = QtWidgets.QGroupBox("1. Video")
        vl = QtWidgets.QVBoxLayout(gb_video)
        self.btn_cargar = QtWidgets.QPushButton("Cargar video (.mp4, .avi, .mov)")
        self.btn_cargar.clicked.connect(self.cargar_video)
        self.lbl_video = QtWidgets.QLabel("Ningún video cargado")
        self.lbl_video.setWordWrap(True)
        self.lbl_video.setObjectName("info")
        vl.addWidget(self.btn_cargar)
        vl.addWidget(self.lbl_video)
        pl.addWidget(gb_video)

        # Grupo: Parámetros
        gb_param = QtWidgets.QGroupBox("2. Parámetros de análisis")
        fl = QtWidgets.QVBoxLayout(gb_param)

        # Umbral de movimiento
        self.lbl_umbral = QtWidgets.QLabel("Umbral de movimiento: 5 px")
        self.slider_umbral = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_umbral.setMinimum(0)
        self.slider_umbral.setMaximum(50)
        self.slider_umbral.setValue(5)
        self.slider_umbral.setTickInterval(5)
        self.slider_umbral.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider_umbral.valueChanged.connect(self._actualizar_umbral)
        ayuda_umbral = QtWidgets.QLabel(
            "Desplazamiento mínimo (px) para contar como movimiento real "
            "y no movimiento de apéndices."
        )
        ayuda_umbral.setWordWrap(True)
        ayuda_umbral.setObjectName("ayuda")
        fl.addWidget(self.lbl_umbral)
        fl.addWidget(self.slider_umbral)
        fl.addWidget(ayuda_umbral)

        # Sensibilidad de detección
        fl.addSpacing(6)
        self.lbl_sensibilidad = QtWidgets.QLabel("Sensibilidad de detección: 10")
        self.slider_sensibilidad = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_sensibilidad.setMinimum(1)
        self.slider_sensibilidad.setMaximum(40)
        self.slider_sensibilidad.setValue(10)
        self.slider_sensibilidad.valueChanged.connect(
            lambda v: self.lbl_sensibilidad.setText(f"Sensibilidad de detección: {v}")
        )
        ayuda_sens = QtWidgets.QLabel(
            "Mayor valor = detecta objetos con menor contraste sobre el fondo."
        )
        ayuda_sens.setWordWrap(True)
        ayuda_sens.setObjectName("ayuda")
        fl.addWidget(self.lbl_sensibilidad)
        fl.addWidget(self.slider_sensibilidad)
        fl.addWidget(ayuda_sens)

        pl.addWidget(gb_param)

        # Grupo: Calibración
        gb_cal = QtWidgets.QGroupBox("3. Distancia / velocidad real (opcional)")
        cl = QtWidgets.QVBoxLayout(gb_cal)
        self.chk_calibrar = QtWidgets.QCheckBox(
            "Calcular distancia y velocidad en unidad real"
        )
        self.chk_calibrar.stateChanged.connect(self._toggle_calibracion)
        cl.addWidget(self.chk_calibrar)

        form = QtWidgets.QFormLayout()
        self.txt_pixeles = QtWidgets.QLineEdit("100")
        self.txt_valor = QtWidgets.QLineEdit("1")
        self.txt_unidad = QtWidgets.QLineEdit("mm")
        self.txt_pixeles.setValidator(QtGui.QDoubleValidator(0.0001, 1e9, 4))
        self.txt_valor.setValidator(QtGui.QDoubleValidator(0.0001, 1e9, 6))
        form.addRow("Píxeles:", self.txt_pixeles)
        form.addRow("equivalen a:", self.txt_valor)
        form.addRow("Unidad:", self.txt_unidad)
        cl.addLayout(form)
        self.lbl_cal_ejemplo = QtWidgets.QLabel("Ejemplo: 100 píxeles = 1 mm")
        self.lbl_cal_ejemplo.setObjectName("ayuda")
        cl.addWidget(self.lbl_cal_ejemplo)
        self._toggle_calibracion()
        pl.addWidget(gb_cal)

        # Grupo: Salida
        gb_sal = QtWidgets.QGroupBox("4. Carpeta de salida")
        sl = QtWidgets.QVBoxLayout(gb_sal)
        self.btn_carpeta = QtWidgets.QPushButton("Seleccionar carpeta de salida")
        self.btn_carpeta.clicked.connect(self.seleccionar_carpeta)
        self.lbl_carpeta = QtWidgets.QLabel("Ninguna carpeta seleccionada")
        self.lbl_carpeta.setWordWrap(True)
        self.lbl_carpeta.setObjectName("info")
        sl.addWidget(self.btn_carpeta)
        sl.addWidget(self.lbl_carpeta)
        pl.addWidget(gb_sal)

        # Botones de acción
        self.btn_iniciar = QtWidgets.QPushButton("▶ Iniciar análisis")
        self.btn_iniciar.setObjectName("primario")
        self.btn_iniciar.clicked.connect(self.iniciar_analisis)
        self.btn_cancelar = QtWidgets.QPushButton("Cancelar")
        self.btn_cancelar.clicked.connect(self.cancelar_analisis)
        self.btn_cancelar.setEnabled(False)
        pl.addWidget(self.btn_iniciar)
        pl.addWidget(self.btn_cancelar)

        pl.addStretch(1)
        layout.addWidget(panel)

        # ---- Panel derecho: visualización y resultados ----
        derecho = QtWidgets.QVBoxLayout()

        self.visor = QtWidgets.QLabel("La visualización del video aparecerá aquí")
        self.visor.setAlignment(QtCore.Qt.AlignCenter)
        self.visor.setObjectName("visor")
        self.visor.setMinimumSize(640, 420)
        self.visor.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        derecho.addWidget(self.visor, stretch=3)

        self.barra = QtWidgets.QProgressBar()
        self.barra.setValue(0)
        derecho.addWidget(self.barra)

        self.lbl_estado = QtWidgets.QLabel("Listo.")
        self.lbl_estado.setObjectName("estado")
        derecho.addWidget(self.lbl_estado)

        gb_res = QtWidgets.QGroupBox("Resultados / Estadísticas")
        rl = QtWidgets.QVBoxLayout(gb_res)
        self.txt_resultados = QtWidgets.QPlainTextEdit()
        self.txt_resultados.setReadOnly(True)
        self.txt_resultados.setObjectName("resultados")
        rl.addWidget(self.txt_resultados)
        derecho.addWidget(gb_res, stretch=2)

        layout.addLayout(derecho, stretch=1)

    # -------------------------------------------------------------- acciones
    def _actualizar_umbral(self, valor: int) -> None:
        self.lbl_umbral.setText(f"Umbral de movimiento: {valor} px")

    def _toggle_calibracion(self) -> None:
        activo = self.chk_calibrar.isChecked()
        for w in (self.txt_pixeles, self.txt_valor, self.txt_unidad):
            w.setEnabled(activo)

    def cargar_video(self) -> None:
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Seleccionar video",
            "",
            "Videos (*.mp4 *.avi *.mov);;Todos los archivos (*)",
        )
        if not ruta:
            return
        if not validar_formato(ruta):
            self._mensaje_error(
                "Formato no soportado",
                "Solo se admiten archivos .mp4, .avi y .mov.",
            )
            return
        try:
            info = leer_info_video(ruta)
        except ErrorVideo as e:
            self._mensaje_error("Error al abrir el video", str(e))
            return
        self.ruta_video = ruta
        self.info_video = info
        self.lbl_video.setText(
            f"<b>{os.path.basename(ruta)}</b><br>"
            f"{info.ancho}x{info.alto} px · {info.fps:.1f} FPS · "
            f"{info.total_frames} frames · {info.duracion:.1f} s"
        )
        # Mostrar primer frame
        self._mostrar_primer_frame(ruta)
        # Sugerir carpeta de salida junto al video
        if not self.carpeta_salida:
            sugerida = os.path.join(os.path.dirname(ruta), "rastreo_salidas")
            self.carpeta_salida = sugerida
            self.lbl_carpeta.setText(sugerida)

    def _mostrar_primer_frame(self, ruta: str) -> None:
        cap = cv2.VideoCapture(ruta)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            self._mostrar_frame(frame)

    def seleccionar_carpeta(self) -> None:
        carpeta = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de salida"
        )
        if carpeta:
            self.carpeta_salida = carpeta
            self.lbl_carpeta.setText(carpeta)

    def _construir_calibracion(self) -> Optional[Calibracion]:
        if not self.chk_calibrar.isChecked():
            return Calibracion(activa=False)
        try:
            pix = float(self.txt_pixeles.text().replace(",", "."))
            val = float(self.txt_valor.text().replace(",", "."))
        except ValueError:
            self._mensaje_error(
                "Calibración inválida",
                "Ingrese valores numéricos válidos para la calibración.",
            )
            return None
        if pix <= 0 or val <= 0:
            self._mensaje_error(
                "Calibración inválida",
                "Los valores de calibración deben ser mayores que cero.",
            )
            return None
        unidad = self.txt_unidad.text().strip() or "u"
        return Calibracion(pixeles=pix, valor_real=val, unidad=unidad, activa=True)

    def _construir_parametros(self) -> ParametrosDeteccion:
        c = self.slider_sensibilidad.value()
        return ParametrosDeteccion(constante_c=c)

    def iniciar_analisis(self) -> None:
        if not self.ruta_video:
            self._mensaje_error("Falta el video", "Primero cargue un video.")
            return
        if not self.carpeta_salida:
            self.seleccionar_carpeta()
            if not self.carpeta_salida:
                return
        calibracion = self._construir_calibracion()
        if calibracion is None:
            return
        try:
            os.makedirs(self.carpeta_salida, exist_ok=True)
        except OSError as e:
            self._mensaje_error(
                "Carpeta inválida", f"No se pudo crear la carpeta de salida:\n{e}"
            )
            return

        self.txt_resultados.clear()
        self.barra.setValue(0)
        self._bloquear_controles(True)

        self.hilo = HiloProcesamiento(
            self.ruta_video,
            float(self.slider_umbral.value()),
            calibracion,
            self._construir_parametros(),
            self.carpeta_salida,
        )
        self.hilo.frame_listo.connect(self._on_frame)
        self.hilo.progreso.connect(self._on_progreso)
        self.hilo.estado.connect(self.lbl_estado.setText)
        self.hilo.terminado.connect(self._on_terminado)
        self.hilo.error.connect(self._on_error)
        self.hilo.start()

    def cancelar_analisis(self) -> None:
        if self.hilo and self.hilo.isRunning():
            self.hilo.cancelar()
            self.lbl_estado.setText("Cancelando...")

    # -------------------------------------------------------------- callbacks
    def _on_frame(self, indice, total, overlay, deteccion) -> None:
        # Mostrar solo algunos frames para fluidez
        if total <= 0 or indice % 2 == 0:
            self._mostrar_frame(overlay)

    def _on_progreso(self, actual, total) -> None:
        if total > 0:
            self.barra.setMaximum(total)
            self.barra.setValue(actual)
        else:
            self.barra.setMaximum(0)

    def _on_terminado(self, resultado: ResultadoAnalisis, info: InfoVideo, ultimo_frame) -> None:
        self.resultado = resultado
        self._bloquear_controles(False)
        self.barra.setMaximum(max(1, info.total_frames))
        self.barra.setValue(info.total_frames)
        self.lbl_estado.setText("Análisis completado.")
        self._mostrar_resumen(resultado)
        if ultimo_frame is not None:
            self._mostrar_frame(ultimo_frame)
        QtWidgets.QMessageBox.information(
            self,
            "Análisis completado",
            "El análisis finalizó correctamente.\n\nLas salidas se guardaron en:\n"
            f"{self.carpeta_salida}",
        )

    def _on_error(self, mensaje: str) -> None:
        self._bloquear_controles(False)
        self.lbl_estado.setText("Error durante el análisis.")
        self._mensaje_error("Error", mensaje)

    # ---------------------------------------------------------------- helpers
    def _bloquear_controles(self, procesando: bool) -> None:
        self.btn_iniciar.setEnabled(not procesando)
        self.btn_cargar.setEnabled(not procesando)
        self.btn_carpeta.setEnabled(not procesando)
        self.btn_cancelar.setEnabled(procesando)

    def _mostrar_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        pix = pix.scaled(
            self.visor.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.visor.setPixmap(pix)

    def _mostrar_resumen(self, r: ResultadoAnalisis) -> None:
        cal = r.calibracion
        lineas = [
            f"FPS: {r.fps:.2f}   |   Frames: {r.frames_totales}  (detectados: {r.frames_detectados})",
            f"Tiempo total:        {r.tiempo_total:.2f} s",
            f"  En movimiento:     {r.tiempo_movimiento:.2f} s",
            f"  Estático:          {r.tiempo_estatico:.2f} s",
            "",
            f"Distancia total:     {r.distancia_total_px:.1f} px",
            f"Vel. promedio:       {r.velocidad_promedio_px:.2f} px/s",
            f"Vel. máxima:         {r.velocidad_maxima_px:.2f} px/s",
        ]
        if cal.activa:
            lineas += [
                "",
                f"Calibración: {cal.pixeles:g} px = {cal.valor_real:g} {cal.unidad}",
                f"Distancia total:     {r.distancia_total_real:.3f} {cal.unidad_distancia}",
                f"Vel. promedio:       {r.velocidad_promedio_real:.3f} {cal.unidad_velocidad}",
                f"Vel. máxima:         {r.velocidad_maxima_real:.3f} {cal.unidad_velocidad}",
            ]
        lineas += [
            "",
            f"Sentido de giro predominante: {r.sentido_predominante}",
            f"  Frames CW (horario):      {r.giros_cw}",
            f"  Frames CCW (antihorario): {r.giros_ccw}",
            "",
            "Salidas generadas:",
            "  • <prefijo>_datos.csv",
            "  • <prefijo>_resumen.txt",
            "  • <prefijo>_larga_exposicion.png",
            "  • <prefijo>_mapa_calor.png",
            "  • <prefijo>_grafica_velocidad.png",
        ]
        self.txt_resultados.setPlainText("\n".join(lineas))

    def _mensaje_error(self, titulo: str, mensaje: str) -> None:
        QtWidgets.QMessageBox.critical(self, titulo, mensaje)

    def cerrar(self) -> None:
        """Cancela el hilo activo (llamado al cerrar la aplicación)."""
        if self.hilo and self.hilo.isRunning():
            self.hilo.cancelar()
            self.hilo.wait(2000)
