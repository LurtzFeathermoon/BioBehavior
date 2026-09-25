"""Módulo de análisis de datos del movimiento del cladócero.

Calcula desplazamiento entre frames, clasifica movimiento vs estático,
determina el sentido de giro (CW/CCW) y la velocidad instantánea y
promedio. Soporta calibración de píxeles a unidades reales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import math

from .tracker import DeteccionFrame, Trayectoria


@dataclass
class Calibracion:
    """Conversión de píxeles a unidad real.

    Ejemplo: "100 píxeles = 1 mm" -> pixeles=100, valor_real=1, unidad="mm".
    """

    pixeles: float = 1.0
    valor_real: float = 1.0
    unidad: str = "px"
    activa: bool = False

    @property
    def factor(self) -> float:
        """Unidades reales por píxel."""
        if not self.activa or self.pixeles == 0:
            return 1.0
        return self.valor_real / self.pixeles

    @property
    def unidad_distancia(self) -> str:
        return self.unidad if self.activa else "px"

    @property
    def unidad_velocidad(self) -> str:
        return f"{self.unidad_distancia}/s"

    def convertir(self, valor_px: float) -> float:
        """Convierte un valor en píxeles a unidad real (o lo deja igual)."""
        return valor_px * self.factor


@dataclass
class PuntoAnalizado:
    """Datos calculados para un frame."""

    indice: int
    tiempo: float
    x: Optional[float]
    y: Optional[float]
    desplazamiento_px: float = 0.0      # distancia respecto al frame previo (px)
    velocidad_px: float = 0.0           # px/s
    velocidad_real: float = 0.0         # unidad/s (calibrada)
    estado: str = "estatico"            # "movimiento" | "estatico"
    sentido_giro: str = "indefinido"    # "CW" | "CCW" | "indefinido"
    detectado: bool = False


@dataclass
class ResultadoAnalisis:
    """Resultado agregado de todo el análisis."""

    puntos: List[PuntoAnalizado] = field(default_factory=list)
    fps: float = 30.0
    calibracion: Calibracion = field(default_factory=Calibracion)

    # Métricas agregadas
    tiempo_total: float = 0.0
    tiempo_movimiento: float = 0.0
    tiempo_estatico: float = 0.0
    distancia_total_px: float = 0.0
    velocidad_promedio_px: float = 0.0
    velocidad_maxima_px: float = 0.0
    frames_detectados: int = 0
    frames_totales: int = 0
    sentido_predominante: str = "indefinido"
    giros_cw: int = 0
    giros_ccw: int = 0

    @property
    def distancia_total_real(self) -> float:
        return self.calibracion.convertir(self.distancia_total_px)

    @property
    def velocidad_promedio_real(self) -> float:
        return self.calibracion.convertir(self.velocidad_promedio_px)

    @property
    def velocidad_maxima_real(self) -> float:
        return self.calibracion.convertir(self.velocidad_maxima_px)


class AnalizadorMovimiento:
    """Analiza una trayectoria y produce métricas y series temporales."""

    def __init__(
        self,
        umbral_movimiento: float = 5.0,
        calibracion: Optional[Calibracion] = None,
        ventana_giro: int = 5,
    ) -> None:
        """
        Args:
            umbral_movimiento: desplazamiento en píxeles por encima del cual
                se considera "movimiento" (default 5 px). Permite filtrar
                movimientos de garras/apéndices.
            calibracion: objeto Calibracion para conversión a unidad real.
            ventana_giro: número de puntos usados para estimar el sentido
                de giro instantáneo mediante el producto cruz.
        """
        self.umbral = float(umbral_movimiento)
        self.calibracion = calibracion or Calibracion()
        self.ventana_giro = max(3, int(ventana_giro))

    @staticmethod
    def _distancia(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def _sentido_local(
        self, puntos: List[PuntoAnalizado], i: int
    ) -> str:
        """Estima el sentido de giro alrededor del punto i usando el
        producto cruz de vectores consecutivos.

        En coordenadas de imagen el eje Y crece hacia abajo, por lo que un
        producto cruz positivo (z) corresponde a sentido horario (CW).
        """
        # Buscar tres puntos válidos espaciados
        idxs = [j for j in range(max(0, i - self.ventana_giro), min(len(puntos), i + self.ventana_giro + 1)) if puntos[j].detectado and puntos[j].x is not None]
        if len(idxs) < 3:
            return "indefinido"
        p0 = puntos[idxs[0]]
        p1 = puntos[idxs[len(idxs) // 2]]
        p2 = puntos[idxs[-1]]
        v1x, v1y = p1.x - p0.x, p1.y - p0.y
        v2x, v2y = p2.x - p1.x, p2.y - p1.y
        cruz = v1x * v2y - v1y * v2x
        # Magnitud mínima para evitar ruido
        if abs(cruz) < 1e-3:
            return "indefinido"
        return "CW" if cruz > 0 else "CCW"

    def analizar(self, trayectoria: Trayectoria, fps: float) -> ResultadoAnalisis:
        fps = fps if fps and fps > 0 else 30.0
        dt = 1.0 / fps
        resultado = ResultadoAnalisis(fps=fps, calibracion=self.calibracion)
        resultado.frames_totales = len(trayectoria.detecciones)

        puntos: List[PuntoAnalizado] = []
        prev: Optional[DeteccionFrame] = None

        for det in trayectoria.detecciones:
            p = PuntoAnalizado(
                indice=det.indice,
                tiempo=det.tiempo,
                x=det.x,
                y=det.y,
                detectado=det.detectado,
            )
            if det.x is not None and prev is not None and prev.x is not None:
                desp = self._distancia(prev.x, prev.y, det.x, det.y)
                p.desplazamiento_px = desp
                p.velocidad_px = desp / dt
                p.velocidad_real = self.calibracion.convertir(p.velocidad_px)
                p.estado = "movimiento" if desp > self.umbral else "estatico"
            else:
                p.estado = "estatico"
            puntos.append(p)
            if det.x is not None:
                prev = det

        # Sentido de giro por punto y conteo global
        for i, p in enumerate(puntos):
            if p.estado == "movimiento" and p.detectado:
                p.sentido_giro = self._sentido_local(puntos, i)
                if p.sentido_giro == "CW":
                    resultado.giros_cw += 1
                elif p.sentido_giro == "CCW":
                    resultado.giros_ccw += 1

        # Métricas agregadas
        resultado.puntos = puntos
        resultado.frames_detectados = sum(1 for p in puntos if p.detectado)
        resultado.tiempo_total = len(puntos) * dt
        resultado.tiempo_movimiento = sum(
            dt for p in puntos if p.estado == "movimiento"
        )
        resultado.tiempo_estatico = resultado.tiempo_total - resultado.tiempo_movimiento
        resultado.distancia_total_px = sum(p.desplazamiento_px for p in puntos)

        velocidades = [p.velocidad_px for p in puntos if p.estado == "movimiento"]
        if velocidades:
            resultado.velocidad_promedio_px = sum(velocidades) / len(velocidades)
            resultado.velocidad_maxima_px = max(velocidades)

        if resultado.giros_cw == 0 and resultado.giros_ccw == 0:
            resultado.sentido_predominante = "indefinido"
        elif resultado.giros_cw >= resultado.giros_ccw:
            resultado.sentido_predominante = "CW"
        else:
            resultado.sentido_predominante = "CCW"

        return resultado
