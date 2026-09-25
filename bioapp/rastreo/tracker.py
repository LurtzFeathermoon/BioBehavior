"""Módulo de seguimiento y detección de cladóceros.

Detecta el objeto más oscuro sobre fondo claro (retroiluminación de
microscopio) usando umbralización adaptativa y rastrea su posición
frame por frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DeteccionFrame:
    """Resultado de detección en un solo frame."""

    indice: int                       # Índice del frame (0-based)
    tiempo: float                     # Tiempo en segundos
    x: Optional[float]                # Centroide X en píxeles (None si no se detecta)
    y: Optional[float]                # Centroide Y en píxeles
    area: float = 0.0                 # Área del contorno detectado (píxeles²)
    detectado: bool = False           # True si se encontró el cladócero


@dataclass
class ParametrosDeteccion:
    """Parámetros configurables del detector."""

    desenfoque: int = 5               # Tamaño kernel de desenfoque gaussiano (impar)
    block_size: int = 51              # Tamaño de bloque del threshold adaptativo (impar)
    constante_c: int = 10             # Constante C sustraída en el threshold adaptativo
    area_minima: int = 15             # Área mínima del contorno para considerarlo válido
    area_maxima: int = 100000         # Área máxima del contorno
    usar_adaptativo: bool = True      # Usar threshold adaptativo (True) o Otsu (False)
    radio_busqueda: float = 60.0      # Radio (px) para asociar candidatos entre frames y dar continuidad
    frames_perdido_max: int = 15      # Frames sin candidato cercano antes de soltar una pista/objetivo
    movimiento_min_pista: float = 0.5  # Desplazamiento promedio (px/frame) mínimo para considerar una pista "en movimiento"


@dataclass
class _Pista:
    """Pista interna de un candidato a través de los frames (multi-objeto)."""

    id: int
    x: float
    y: float
    area: float
    frames_vistos: int = 1
    frames_perdido: int = 0
    desplazamiento_acumulado: float = 0.0

    @property
    def movimiento_promedio(self) -> float:
        if self.frames_vistos <= 1:
            return 0.0
        return self.desplazamiento_acumulado / (self.frames_vistos - 1)


class DetectorCladocero:
    """Detecta el cladócero (objeto más oscuro) en un frame.

    Estrategia:
        1. Convertir a escala de grises.
        2. Suavizar con desenfoque gaussiano para reducir ruido.
        3. Umbralización adaptativa para resaltar objetos oscuros sobre
           fondo claro (THRESH_BINARY_INV: lo oscuro queda en blanco).
        4. Operaciones morfológicas para limpiar la máscara.
        5. Seguir cada candidato como una "pista" a través de los frames
           y elegir como objetivo aquel con mayor desplazamiento promedio
           real (no solo el de mayor área), descartando así elementos
           estáticos como sombras o suciedad del microscopio que antes
           "ganaban" por tener más área o contraste.
    """

    def __init__(self, parametros: Optional[ParametrosDeteccion] = None) -> None:
        self.parametros = parametros or ParametrosDeteccion()
        self._pistas: dict = {}
        self._siguiente_id: int = 0
        self._objetivo_id: Optional[int] = None

    def reset(self) -> None:
        """Reinicia el estado de seguimiento (pistas y objetivo actual)."""
        self._pistas = {}
        self._siguiente_id = 0
        self._objetivo_id = None

    def _kernel_impar(self, valor: int, minimo: int = 1) -> int:
        valor = max(minimo, int(valor))
        if valor % 2 == 0:
            valor += 1
        return valor

    def crear_mascara(self, frame: np.ndarray) -> np.ndarray:
        """Devuelve la máscara binaria del objeto oscuro."""
        if frame.ndim == 3:
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gris = frame

        k = self._kernel_impar(self.parametros.desenfoque, 1)
        if k > 1:
            gris = cv2.GaussianBlur(gris, (k, k), 0)

        if self.parametros.usar_adaptativo:
            block = self._kernel_impar(self.parametros.block_size, 3)
            mascara = cv2.adaptiveThreshold(
                gris,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block,
                self.parametros.constante_c,
            )
        else:
            # Otsu invertido: lo oscuro se vuelve blanco
            _, mascara = cv2.threshold(
                gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Limpieza morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mascara

    def _candidatos(self, mascara: np.ndarray) -> List[Tuple[np.ndarray, float, float, float]]:
        """Extrae (contorno, área, cx, cy) de los contornos válidos por área."""
        contornos = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        candidatos = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area < self.parametros.area_minima or area > self.parametros.area_maxima:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            candidatos.append((c, area, cx, cy))
        return candidatos

    def _actualizar_pistas(self, candidatos: List[Tuple[np.ndarray, float, float, float]]) -> set:
        """Asocia los candidatos del frame actual con las pistas existentes
        (vecino más cercano dentro de radio_busqueda) para ir acumulando el
        desplazamiento real de cada una. Devuelve el conjunto de ids de
        pistas que fueron emparejadas (detectadas) en este frame."""
        radio = self.parametros.radio_busqueda

        pares = []
        for idx_c, (_c, _area, cx, cy) in enumerate(candidatos):
            for pid, pista in self._pistas.items():
                d = math.hypot(cx - pista.x, cy - pista.y)
                if d <= radio:
                    pares.append((d, idx_c, pid))
        pares.sort(key=lambda t: t[0])

        asignacion_candidato = {}
        usados_p = set()
        for d, idx_c, pid in pares:
            if idx_c in asignacion_candidato or pid in usados_p:
                continue
            asignacion_candidato[idx_c] = pid
            usados_p.add(pid)

        nuevas_pistas = {}
        actualizadas = set()
        for idx_c, (_c, area, cx, cy) in enumerate(candidatos):
            pid = asignacion_candidato.get(idx_c)
            if pid is not None:
                pista = self._pistas[pid]
                pista.desplazamiento_acumulado += math.hypot(cx - pista.x, cy - pista.y)
                pista.frames_vistos += 1
                pista.frames_perdido = 0
                pista.x, pista.y, pista.area = cx, cy, area
            else:
                pid = self._siguiente_id
                self._siguiente_id += 1
                pista = _Pista(id=pid, x=cx, y=cy, area=area)
            nuevas_pistas[pid] = pista
            actualizadas.add(pid)

        for pid, pista in self._pistas.items():
            if pid in nuevas_pistas:
                continue
            pista.frames_perdido += 1
            if pista.frames_perdido <= self.parametros.frames_perdido_max:
                nuevas_pistas[pid] = pista

        self._pistas = nuevas_pistas
        return actualizadas

    def _elegir_objetivo(self, actualizadas: set) -> Optional[int]:
        """Elige el objetivo entre las pistas conocidas dando prioridad a la
        evidencia real de movimiento (desplazamiento promedio por frame) en
        vez de al área. Mantiene el objetivo actual mientras siga siendo
        detectado y no haya otra pista mostrando movimiento significativo
        que él no tenga; de lo contrario se reevalúa y se cambia al
        candidato más móvil. Si ninguna pista se mueve lo suficiente, se
        recurre a la de mayor área como último recurso."""
        if not self._pistas:
            return None

        moviles = {
            pid for pid, p in self._pistas.items()
            if p.movimiento_promedio >= self.parametros.movimiento_min_pista
        }

        actual = self._objetivo_id
        if (
            actual is not None
            and actual in self._pistas
            and actual in actualizadas
            and (not moviles or actual in moviles)
        ):
            return actual

        fuente_ids = moviles if moviles else set(self._pistas.keys())
        mejor = max(
            (self._pistas[pid] for pid in fuente_ids),
            key=lambda p: (p.movimiento_promedio, p.area),
        )
        return mejor.id

    def detectar(
        self, frame: np.ndarray, indice: int, tiempo: float
    ) -> Tuple[DeteccionFrame, np.ndarray]:
        """Detecta el cladócero en un frame.

        En lugar de elegir siempre el contorno más grande, se acumula el
        desplazamiento real de cada candidato a través de los frames
        (pistas) y se fija como objetivo al que muestra movimiento
        significativo, evitando engancharse a elementos estáticos (sombras,
        suciedad del microscopio, viñeteado) que solo destacan por área o
        contraste. Una vez fijado el objetivo, se le da continuidad
        mientras siga siendo detectado o esté dentro de la tolerancia de
        frames perdidos.

        Devuelve la detección y la máscara binaria (para depuración/overlay).
        """
        mascara = self.crear_mascara(frame)
        candidatos = self._candidatos(mascara)
        actualizadas = self._actualizar_pistas(candidatos)

        self._objetivo_id = self._elegir_objetivo(actualizadas)

        if self._objetivo_id is None or self._objetivo_id not in actualizadas:
            return DeteccionFrame(indice, tiempo, None, None, 0.0, False), mascara

        pista = self._pistas[self._objetivo_id]
        return (
            DeteccionFrame(indice, tiempo, pista.x, pista.y, pista.area, True),
            mascara,
        )


class Trayectoria:
    """Acumula las detecciones para formar la trayectoria completa."""

    def __init__(self) -> None:
        self.detecciones: List[DeteccionFrame] = []

    def agregar(self, deteccion: DeteccionFrame) -> None:
        self.detecciones.append(deteccion)

    def __len__(self) -> int:
        return len(self.detecciones)

    @property
    def puntos_validos(self) -> List[DeteccionFrame]:
        """Solo las detecciones donde el cladócero fue encontrado."""
        return [d for d in self.detecciones if d.detectado]

    def coordenadas(self) -> List[Tuple[float, float]]:
        return [(d.x, d.y) for d in self.puntos_validos]

    def rellenar_huecos(self) -> None:
        """Interpola posiciones faltantes entre detecciones válidas.

        Útil cuando el cladócero se pierde uno o pocos frames. Modifica
        las detecciones in-place asignando x/y interpolados (manteniendo
        detectado=False para que el análisis sepa que fue interpolado).
        """
        n = len(self.detecciones)
        # Índices con detección válida
        validos = [i for i, d in enumerate(self.detecciones) if d.detectado]
        if len(validos) < 2:
            return
        for a, b in zip(validos, validos[1:]):
            if b - a <= 1:
                continue
            da, db = self.detecciones[a], self.detecciones[b]
            for i in range(a + 1, b):
                t = (i - a) / (b - a)
                self.detecciones[i].x = da.x + (db.x - da.x) * t
                self.detecciones[i].y = da.y + (db.y - da.y) * t
