"""Hoja de estilos única de BioBehaviour.

Todas las pestañas comparten esta apariencia (derivada de la interfaz de
DaphniaTracker): fondo claro, tarjetas blancas con grupos numerados,
botón primario verde y visor oscuro.
"""

QSS = """
QMainWindow { background: #f4f6f8; }
QWidget#pagina { background: #f4f6f8; }
QTabWidget::pane { border: none; background: #f4f6f8; }
QTabBar::tab {
    background: #e3e9ee; color: #2c3e50;
    padding: 9px 22px; margin-right: 4px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-weight: bold; font-size: 13px;
}
QTabBar::tab:hover { background: #d5dee6; }
QTabBar::tab:selected { background: #16607a; color: white; }
QGroupBox {
    font-weight: bold; border: 1px solid #c9d2da;
    border-radius: 6px; margin-top: 10px; padding: 8px;
    background: #ffffff;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #2c3e50; }
QLabel { color: #2c3e50; }
QLabel#subtitulo { font-size: 12px; color: #5a6b7b; }
QLabel#info { color: #34495e; font-size: 11px; }
QLabel#ayuda { color: #7f8c8d; font-size: 10px; font-style: italic; }
QLabel#estado { color: #16607a; font-weight: bold; }
QLabel#visor { background: #1b2733; color: #8fa3b3; border-radius: 6px; }
QPushButton {
    padding: 8px; border-radius: 5px; border: 1px solid #b7c3cd;color: #34495e;
    background: #eef2f5;
}
QPushButton:hover { background: #e1e8ee; }
QPushButton#primario {
    background: #16a085; color: white; font-weight: bold; border: none;
}
QPushButton#primario:hover { background: #138a72; }
QPushButton:disabled { color: #16607a; background: #b7c3cd; }
QPlainTextEdit#resultados { font-family: 'Consolas','DejaVu Sans Mono',monospace; font-size: 11px; }
QProgressBar { border: 1px solid #b7c3cd; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #16a085; }
QCheckBox { color: #2c3e50; }
QLineEdit {
    padding: 4px 6px; border: 1px solid #b7c3cd; border-radius: 4px;
    background: #ffffff; color: #2c3e50;
}
QLineEdit:focus { border-color: #16a085; }
QComboBox {
    padding: 4px 6px; border: 1px solid #b7c3cd; border-radius: 4px;
    background: #ffffff; color: #2c3e50;
}
QComboBox:focus { border-color: #16a085; }
QSlider::groove:horizontal { height: 6px; background: #d5dee6; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 14px; margin: -5px 0; border-radius: 7px;
    background: #16a085;
}
QRadioButton { color: #2c3e50; }
"""
