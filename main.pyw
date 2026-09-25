#!/usr/bin/env python3
"""Punto de entrada de BioBehaviour (aplicación unificada).

Ejecutar con:
    python main.py

O por doble clic en main.pyw (sin ventana de consola).
"""

import sys


def main() -> int:
    try:
        from bioapp.ventana import VentanaPrincipal
    except ImportError as e:
        print("Error: faltan dependencias.")
        print(f"Detalle: {e}")
        print("Instale con: pip install -r requirements.txt")
        return 1

    from PySide6.QtWidgets import QApplication

    import ctypes

    from bioapp import APP_TITLE
    from bioapp.estilo import QSS

    if sys.platform.startswith("win"):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "BioBehaviourApp"
        )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setStyleSheet(QSS)
    ventana = VentanaPrincipal()
    ventana.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
