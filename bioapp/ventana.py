"""Ventana principal de BioBehaviour: área de trabajo con pestañas."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from . import APP_SUBTITLE, APP_TITLE


class VentanaPrincipal(QMainWindow):
    """Ventana con las tres herramientas de análisis en pestañas."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} — Análisis de comportamiento animal")
        self.resize(1280, 820)

        self.paginas = []

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(False)

        from .cardiaco.gui import PanelCardiaco
        from .rastreo.gui import PanelRastreo
        from .sombra.gui import PanelSombra

        self.panel_cardiaco = PanelCardiaco()
        self.panel_rastreo = PanelRastreo()
        self.panel_sombra = PanelSombra()

        tabs.addTab(self.panel_cardiaco, "1 · Pulso cardíaco")
        tabs.addTab(self.panel_rastreo, "2 · Rastreo de movimiento")
        tabs.addTab(self.panel_sombra, "3 · Mapa de calor")
        self.setCentralWidget(tabs)
        self.paginas = [self.panel_cardiaco, self.panel_rastreo, self.panel_sombra]

        self.statusBar().showMessage(f"{APP_SUBTITLE}")

    def cerrar_paneles(self) -> None:
        for p in self.paginas:
            p.cerrar()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.cerrar_paneles()
        event.accept()
