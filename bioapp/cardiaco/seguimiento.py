"""Seguimiento automático de la región de interés (ROI) mediante una
máscara de segmentación del organismo.

Cuando el organismo se desplaza o gira sobre sí mismo, una ROI fija en
coordenadas de imagen puede perder el corazón. Este módulo segmenta la
silueta del organismo por contraste local (no por diferencia con un
fondo temporal: si el organismo apenas se mueve, un fondo "promedio"
del video terminaría absorbiéndolo y la segmentación por movimiento
dejaría de funcionar). En una ventana centrada en la última posición
conocida se separa el organismo del entorno inmediato mediante un
umbral de Otsu, se calcula el centroide y la orientación del cuerpo, y
se reubica/rota la ROI para que mantenga su posición relativa sobre el
organismo.

Es intencionalmente genérico (no específico de la daphnia): funciona
sobre cualquier silueta con contraste frente a su entorno, así que
puede usarse con otras especies, con la limitación obvia de que solo
sigue el área originalmente seleccionada por el usuario (no vuelve a
"buscar" el corazón dentro del cuerpo).
"""

from __future__ import annotations

import cv2
import numpy as np

ELONGACION_MINIMA = 0.20
SUAVIZADO_POSICION = 0.35   # peso del frame nuevo en el suavizado de (cx, cy)
SUAVIZADO_ANGULO = 0.25     # peso del frame nuevo en el suavizado del ángulo
MAX_GIRO_POR_FRAME = 8.0    # grados; limita saltos bruscos de ángulo por ruido
FACTOR_VENTANA = 3.0        # tamaño de la ventana de búsqueda local vs. el ROI
VENTANA_MINIMA = 70


def construir_fondo(cap, n_muestras=25):
    """Se conserva por compatibilidad, pero ya no se usa para segmentar:
    un fondo "promedio" del video falla cuando el organismo está quieto
    (se absorbe en el fondo). Devuelve None sin costo."""
    return None


def _segmentar_en_ventana(gray_win, area_esperada=None):
    """Segmenta el objeto de mayor contraste dentro de una ventana
    local probando ambas polaridades (organismo más oscuro o más claro
    que su entorno) y elige el contorno más plausible: centrado en la
    ventana, de tamaño razonable y, si se conoce, de área similar a la
    del frame anterior (continuidad)."""
    h, w = gray_win.shape[:2]
    area_win = float(w * h)
    if area_win <= 0:
        return None

    blur = cv2.GaussianBlur(gray_win, (5, 5), 0)
    _, mask_a = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_b = cv2.bitwise_not(mask_a)

    kernel = np.ones((3, 3), np.uint8)
    centro = (w / 2.0, h / 2.0)

    mejor_contorno = None
    mejor_score = -np.inf

    for mask in (mask_a, mask_b):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.02 * area_win or area > 0.85 * area_win:
                continue

            m = cv2.moments(c)
            if m["m00"] == 0:
                continue

            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            dist = np.hypot(cx - centro[0], cy - centro[1])

            score = -dist / max(w, h)
            if area_esperada:
                score -= 0.8 * abs(area - area_esperada) / area_esperada

            if score > mejor_score:
                mejor_score = score
                mejor_contorno = c

    return mejor_contorno


def segmentar_organismo(gray, fondo, min_area_fraccion=0.01):
    """Compatibilidad hacia atrás: segmenta sobre el frame completo
    (sin ventana local ni continuidad de área)."""
    return _segmentar_en_ventana(gray)


def pose_desde_contorno(contour, angulo_previo=None):
    """Centroide (cx, cy), ángulo del eje principal (grados) y una
    bandera de confiabilidad del ángulo. El ángulo de una elipse solo
    está definido módulo 180°; si se da un ángulo previo se elige la
    rama continua más cercana para evitar saltos de 180°.

    Cuando la silueta es casi circular (poco alargada), el eje
    principal no está bien definido y el ángulo calculado es puro
    ruido: en ese caso se marca como no confiable (angulo_valido=False)
    para que el llamador conserve el ángulo anterior en vez de girar la
    ROI aleatoriamente."""
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]

    anisotropia = np.hypot(mu20 - mu02, 2.0 * mu11)
    escala = mu20 + mu02 + 1e-9
    elongacion = anisotropia / escala
    angulo_valido = elongacion >= ELONGACION_MINIMA

    angulo = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02))

    if angulo_previo is not None:
        candidatos = [angulo, angulo + 180.0, angulo - 180.0]
        angulo = min(candidatos, key=lambda a: abs(a - angulo_previo))

    return float(cx), float(cy), float(angulo), bool(angulo_valido)


class SeguidorROI:
    """Sigue una ROI rígida sobre el organismo, compensando traslación
    y rotación mediante segmentación por contraste local (no requiere
    que el organismo se mueva). Si en algún frame no se puede segmentar
    el organismo, la ROI conserva su última posición conocida (no
    salta ni se pierde).

    El ángulo solo se actualiza cuando la silueta es lo bastante
    alargada como para que el eje principal esté bien definido, y se
    suaviza en el tiempo con un límite de giro máximo por frame para
    evitar que el ruido de segmentación haga "girar" la ROI de forma
    aleatoria en videos donde el organismo está quieto."""

    def __init__(self, fondo=None):
        self.pose_ref = None
        self.centro_roi_ref = None
        self.pose_prev = None
        self.ultima_roi = None
        self._cx_suave = None
        self._cy_suave = None
        self._ang_suave = None
        self._area_prev = None
        self._tam_ventana = None

    def _ventana(self, frame_bgr, cx, cy):
        fh, fw = frame_bgr.shape[:2]
        lado = int(max(VENTANA_MINIMA, self._tam_ventana))
        lado = min(lado, fh, fw)

        x0 = int(round(cx - lado / 2))
        y0 = int(round(cy - lado / 2))
        x0 = max(0, min(x0, fw - lado))
        y0 = max(0, min(y0, fh - lado))

        recorte = frame_bgr[y0:y0 + lado, x0:x0 + lado]
        return recorte, x0, y0

    def _segmentar(self, frame_bgr, cx, cy):
        ventana, x0, y0 = self._ventana(frame_bgr, cx, cy)
        if ventana.size == 0:
            return None

        gray = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)
        contorno_local = _segmentar_en_ventana(gray, area_esperada=self._area_prev)
        if contorno_local is None:
            return None

        contorno = contorno_local.copy()
        contorno[:, 0, 0] += x0
        contorno[:, 0, 1] += y0
        return contorno

    def inicializar(self, frame_bgr, roi):
        x, y, w, h = roi
        self.ultima_roi = (float(x + w / 2.0), float(y + h / 2.0), w, h, 0.0)

        # La ROI suele quedar cerca del borde del cuerpo (p. ej. el
        # corazón no está en el centro del organismo), así que para
        # detectar el cuerpo completo se busca primero en una ventana
        # amplia centrada en la ROI, en vez de una ventana del tamaño
        # de la ROI (que capturaría solo un recorte parcial y sesgado
        # del cuerpo).
        fh, fw = frame_bgr.shape[:2]
        self._tam_ventana = min(fh, fw, max(8 * max(w, h), 220))
        contorno = self._segmentar(frame_bgr, x + w / 2.0, y + h / 2.0)
        if contorno is None:
            return False

        pose = pose_desde_contorno(contorno)
        if pose is None:
            return False

        # Con el cuerpo ya localizado, se fija un tamaño de ventana de
        # seguimiento proporcional al tamaño real del cuerpo (y no al
        # de la ROI), para que las ventanas de los frames siguientes
        # contengan el cuerpo completo.
        (_, _), (lado_a, lado_b), _ = cv2.minAreaRect(contorno)
        largo_cuerpo = max(lado_a, lado_b, max(w, h))
        self._tam_ventana = min(fh, fw, max(FACTOR_VENTANA * largo_cuerpo, VENTANA_MINIMA))

        cx, cy, ang, _valido = pose
        self.pose_prev = (cx, cy, ang)
        self.pose_ref = (cx, cy, ang)
        self._cx_suave, self._cy_suave, self._ang_suave = cx, cy, ang
        self._area_prev = cv2.contourArea(contorno)
        self.centro_roi_ref = (x + w / 2.0, y + h / 2.0)
        return True

    def actualizar(self, frame_bgr, roi):
        """Devuelve (cx, cy, w, h, angulo_relativo_grados) con la nueva
        posición/rotación de la ROI para este frame."""
        x, y, w, h = roi
        if self.pose_ref is None:
            return self.ultima_roi

        contorno = self._segmentar(frame_bgr, self._cx_suave, self._cy_suave)
        if contorno is None:
            return self.ultima_roi

        # La continuidad de rama (ambigüedad de 180°) se resuelve contra
        # el ángulo ya suavizado, no contra el ángulo crudo del frame
        # anterior: si se mezclan ambas referencias, la rama elegida y
        # el suavizado quedan desincronizados y el ángulo deriva de
        # forma sistemática en vez de quedarse estable.
        pose = pose_desde_contorno(contorno, angulo_previo=self._ang_suave)
        if pose is None:
            return self.ultima_roi

        cx_t, cy_t, ang_t, angulo_valido = pose
        self.pose_prev = (cx_t, cy_t, ang_t)
        self._area_prev = cv2.contourArea(contorno)

        # Suavizado exponencial de la posición.
        self._cx_suave += SUAVIZADO_POSICION * (cx_t - self._cx_suave)
        self._cy_suave += SUAVIZADO_POSICION * (cy_t - self._cy_suave)

        # El ángulo solo se actualiza si la silueta está lo bastante
        # alargada; si no, se conserva el último ángulo suavizado.
        if angulo_valido:
            diff = ang_t - self._ang_suave
            # normalizar al rango [-90, 90] (el ángulo es módulo 180°)
            diff = (diff + 90.0) % 180.0 - 90.0
            diff = max(-MAX_GIRO_POR_FRAME, min(MAX_GIRO_POR_FRAME, diff))
            self._ang_suave += SUAVIZADO_ANGULO * diff

        cx0, cy0, ang0 = self.pose_ref
        cx_roi0, cy_roi0 = self.centro_roi_ref

        delta = self._ang_suave - ang0
        rad = np.radians(delta)
        c, s = np.cos(rad), np.sin(rad)

        dx0, dy0 = cx_roi0 - cx0, cy_roi0 - cy0
        dx_t = c * dx0 - s * dy0
        dy_t = s * dx0 + c * dy0

        self.ultima_roi = (
            self._cx_suave + dx_t,
            self._cy_suave + dy_t,
            w, h, delta,
        )
        return self.ultima_roi


def recortar_rotado(frame_bgr, cx, cy, w, h, angulo_grados):
    """Extrae un recorte w×h centrado en (cx, cy), rotando el contenido
    para compensar el giro respecto a la orientación de referencia.
    Devuelve None si la posición cae fuera del frame."""
    fh, fw = frame_bgr.shape[:2]
    lado = int(np.ceil(np.hypot(w, h))) + 6
    lado = min(lado, fh, fw)
    if lado < 4:
        return None

    x0 = int(round(cx - lado / 2))
    y0 = int(round(cy - lado / 2))
    x0 = max(0, min(x0, fw - lado))
    y0 = max(0, min(y0, fh - lado))

    parche = frame_bgr[y0:y0 + lado, x0:x0 + lado]
    if parche.shape[0] != lado or parche.shape[1] != lado:
        return None

    centro_parche = (cx - x0, cy - y0)
    if abs(angulo_grados) > 1e-6:
        m = cv2.getRotationMatrix2D(centro_parche, angulo_grados, 1.0)
        parche = cv2.warpAffine(parche, m, (lado, lado), flags=cv2.INTER_LINEAR)

    rx0 = int(round(centro_parche[0] - w / 2))
    ry0 = int(round(centro_parche[1] - h / 2))
    rx0 = max(0, min(rx0, lado - w))
    ry0 = max(0, min(ry0, lado - h))

    recorte = parche[ry0:ry0 + h, rx0:rx0 + w]
    if recorte.shape[0] != h or recorte.shape[1] != w:
        return None

    return recorte
