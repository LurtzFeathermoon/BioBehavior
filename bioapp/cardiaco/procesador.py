import cv2
import numpy as np
from scipy.signal import find_peaks

from .seguimiento import SeguidorROI, construir_fondo, recortar_rotado

BPM_MIN = 80
#BPM_MAX = 320  
#BPM_MAX = 360  # detección de pulsos más rápidos
BPM_MAX = 480  # ampliado para no truncar pulsos muy rápidos

BASELINE_SEC_INTENSITY  = 0.70
BASELINE_SEC_MOTION     = 0.50
SMOOTH_SEC_INTENSITY    = 0.06
SMOOTH_SEC_MOTION       = 0.06

DISTANCE_FACTORS        = (0.35, 0.45, 0.55, 0.70)
#DISTANCE_FACTORS        = (0.25, 0.35, 0.45, 0.60) # Latidos más cercanos

PROMINENCE_VALUES       = (0.25, 0.40, 0.60, 0.85)
#PROMINENCE_VALUES       = (0.20, 0.30, 0.45, 0.65) # Detección más sensibles

def _moving_average(x, window):
    x = np.asarray(x, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x.copy()

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def _robust_zscore(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad + 1e-9
    return (x - med) / scale


def _normalize_01(x):
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, 2)
    hi = np.percentile(x, 98)

    if hi - lo < 1e-9:
        xmin = float(np.min(x))
        xmax = float(np.max(x))
        if xmax - xmin < 1e-9:
            return np.zeros_like(x, dtype=np.float32)
        return (x - xmin) / (xmax - xmin)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _estimate_period_acf(signal, fps, bpm_min=BPM_MIN, bpm_max=BPM_MAX):
    x = np.asarray(signal, dtype=np.float32)
    x = x - np.mean(x)

    if len(x) < 10:
        return None, 0.0

    energy = float(np.dot(x, x))
    if energy < 1e-12:
        return None, 0.0

    acf = np.correlate(x, x, mode="full")[len(x) - 1:]
    acf = acf / (acf[0] + 1e-9)

    min_lag = max(1, int(round(fps * 60.0 / bpm_max)))
    max_lag = min(len(acf) - 1, int(round(fps * 60.0 / bpm_min)))

    if max_lag <= min_lag:
        return None, 0.0

    # Ampliamos un poco la ventana de búsqueda hacia ambos lados: un pico
    # justo en el borde de [min_lag, max_lag] no puede detectarse como tal
    # (find_peaks necesita vecinos a los dos lados), lo que hacía perder el
    # periodo fundamental en pulsos muy rápidos (cerca de bpm_max).
    pad = 3
    win_start = max(1, min_lag - pad)
    win_end = min(len(acf) - 1, max_lag + pad)
    search_full = acf[win_start:win_end + 1]
    if len(search_full) < 3:
        return None, 0.0

    peaks_full, props = find_peaks(search_full, prominence=0.02)
    lags_full = win_start + peaks_full
    in_range = (lags_full >= min_lag) & (lags_full <= max_lag)
    peaks = peaks_full[in_range]
    lags = lags_full[in_range]

    search = acf[min_lag:max_lag + 1]

    if len(peaks) == 0:
        idx = int(np.argmax(search))
        lag = min_lag + idx
        strength = float(search[idx])
        return lag, strength

    prominences = props.get("prominences", np.zeros(len(peaks_full), dtype=np.float32))[in_range]
    heights = search_full[peaks]
    max_height = float(np.max(heights))

    # Entre los picos casi igual de fuertes (candidatos a armónicos del
    # periodo real), preferimos el de menor lag: así evitamos que el
    # algoritmo confunda el 2º armónico (mitad del bpm real) con el
    # periodo fundamental, error que a bpm altos truncaba el resultado.
    strong = np.where(heights >= 0.90 * max_height)[0]
    candidate_idxs = strong if len(strong) else np.arange(len(peaks))
    best_idx = int(candidate_idxs[np.argmin(lags[candidate_idxs])])

    lag = int(lags[best_idx])
    strength = float(prominences[best_idx])
    return lag, strength


def _refine_peak_positions(signal, peaks):
    """Interpola de forma sub-frame (parabólica) la posición real de cada
    pico para evitar la cuantización a números enteros de frame, que a
    fps bajos puede truncar el BPM calculado a valores discretos
    (p. ej. exactamente 300 bpm)."""
    x = np.asarray(signal, dtype=np.float64)
    refined = np.asarray(peaks, dtype=np.float64).copy()

    for k, p in enumerate(peaks):
        p = int(p)
        if p <= 0 or p >= len(x) - 1:
            continue

        y0, y1, y2 = x[p - 1], x[p], x[p + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) < 1e-9:
            continue

        delta = 0.5 * (y0 - y2) / denom
        delta = max(-0.5, min(0.5, delta))
        refined[k] = p + delta

    return refined


def _score_peak_train(peaks, prominences, fps, ref_lag, bpm_min=BPM_MIN, bpm_max=BPM_MAX, refined_positions=None):
    if peaks is None or len(peaks) < 4:
        return -np.inf, None, {}

    positions = refined_positions if refined_positions is not None else peaks
    intervals = np.diff(positions).astype(np.float32)
    if len(intervals) < 3:
        return -np.inf, None, {}

    med_interval = float(np.median(intervals))
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))

    if med_interval <= 0 or mean_interval <= 0:
        return -np.inf, None, {}

    bpm = 60.0 * fps / med_interval
    if bpm < bpm_min or bpm > bpm_max:
        return -np.inf, None, {}

    cv = std_interval / (mean_interval + 1e-9)
    prom_med = float(np.median(prominences)) if len(prominences) else 0.0

    score = 0.0
    score -= 2.2 * cv
    score += 0.25 * prom_med
    score += 0.03 * len(peaks)

    if ref_lag is not None and ref_lag > 0:
        rel_err = abs(med_interval - ref_lag) / ref_lag
        score -= 2.5 * rel_err

        if 1.70 * ref_lag <= med_interval <= 2.30 * ref_lag:
            score -= 2.0

        if 0.40 * ref_lag <= med_interval <= 0.65 * ref_lag:
            score -= 1.0

    metrics = {
        "peak_count": int(len(peaks)),
        "median_interval_frames": med_interval,
        "median_interval_sec": med_interval / fps,
        "bpm_from_median_interval": bpm,
        "interval_cv": cv,
        "median_prominence": prom_med,
    }
    return score, bpm, metrics

# ========================= picos señal normal / valles invierte la señal
def _detect_from_work_signal(work_signal, fps, ref_lag, bpm_min=BPM_MIN, bpm_max=BPM_MAX):
    x = np.asarray(work_signal, dtype=np.float32)

    if ref_lag is None or ref_lag <= 0:
        ref_lag = int(round(fps * 60.0 / 180.0))

    best = {
        "score": -np.inf,
        "bpm": None,
        "peaks": None,
        "metrics": {},
    }

    for f in DISTANCE_FACTORS:
        min_dist = max(2, int(round(ref_lag * f)))

        for prom in PROMINENCE_VALUES:
            peaks, props = find_peaks(
                x,
                distance=min_dist,
                prominence=prom
            )

            prominences = props.get("prominences", np.array([], dtype=np.float32))
            refined = _refine_peak_positions(x, peaks) if len(peaks) else peaks
            score, bpm, metrics = _score_peak_train(
                peaks,
                prominences,
                fps,
                ref_lag,
                bpm_min=bpm_min,
                bpm_max=bpm_max,
                refined_positions=refined
            )

            if score > best["score"]:
                best["score"] = score
                best["bpm"] = bpm
                best["peaks"] = peaks
                best["metrics"] = metrics

    return best["peaks"], best["bpm"], best["score"], best["metrics"]

# ========================= procesamiento de intensidad de luz
def _preprocess_intensity(intensity, fps):
    x = np.asarray(intensity, dtype=np.float32)
    smooth_w = max(1, int(round(fps * SMOOTH_SEC_INTENSITY)))
    x_s = _moving_average(x, smooth_w)

    base_w = max(5, int(round(fps * BASELINE_SEC_INTENSITY)))
    baseline = _moving_average(x_s, base_w)
    x_hp = x_s - baseline

    x_z = _robust_zscore(x_hp)
    x_display = _normalize_01(x_z)
    return x_z, x_display

# ========================= procesamiento de intensidad de movimiento
def _preprocess_motion(motion, fps):
    x = np.asarray(motion, dtype=np.float32)
    smooth_w = max(1, int(round(fps * SMOOTH_SEC_MOTION)))
    x_s = _moving_average(x, smooth_w)

    base_w = max(5, int(round(fps * BASELINE_SEC_MOTION)))
    baseline = _moving_average(x_s, base_w)
    x_hp = x_s - baseline

    x_z = _robust_zscore(x_hp)
    x_display = _normalize_01(x_z)
    return x_z, x_display


def _extract_roi_signals(video_path, roi, accelerated, progress_cb, frame_cb, cancel_flag, seguir_automatico=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x, y, w, h = roi

    seguidor = None
    if seguir_automatico:
        fondo = construir_fondo(cap)
        seguidor = SeguidorROI(fondo)

    intensity = []
    motion = []
    prev_gray = None
    frame_idx = 0

    while True:
        if cancel_flag():
            cap.release()
            return None, None, None

        ret, frame = cap.read()
        if not ret:
            break

        roi_actual = roi
        angulo_actual = 0.0
        roi_frame = None

        if seguidor is not None:
            if seguidor.pose_ref is None and seguidor.pose_prev is None:
                seguidor.inicializar(frame, roi)
                cx, cy, w0, h0, ang = seguidor.ultima_roi
            else:
                cx, cy, w0, h0, ang = seguidor.actualizar(frame, roi)

            roi_actual = (cx, cy, w0, h0)
            angulo_actual = ang
            roi_frame = recortar_rotado(frame, cx, cy, w0, h0, ang)

            if roi_frame is None:
                # Recorte rotado fuera de límites: se usa un recorte
                # simple centrado en la misma posición como respaldo.
                fh, fw = frame.shape[:2]
                xi = int(max(0, min(round(cx - w0 / 2), fw - w0)))
                yi = int(max(0, min(round(cy - h0 / 2), fh - h0)))
                roi_frame = frame[yi:yi + int(h0), xi:xi + int(w0)]
        else:
            roi_frame = frame[y:y+h, x:x+w]

        if roi_frame is None or roi_frame.size == 0:
            cap.release()
            raise RuntimeError("El ROI está fuera de los límites del frame")

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        intensity.append(float(np.mean(gray)))

        if prev_gray is None:
            motion.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            motion.append(float(np.mean(diff)))

        prev_gray = gray

        frame_idx += 1
        if progress_cb and total_frames > 0:
            progress_cb(int(100 * frame_idx / total_frames))

        if not accelerated and frame_cb:
            if seguidor is not None:
                cx, cy, w0, h0 = roi_actual
                frame_cb(frame.copy(), (cx, cy, w0, h0, angulo_actual))
            else:
                frame_cb(frame.copy(), roi)

    cap.release()
    return np.asarray(intensity, dtype=np.float32), np.asarray(motion, dtype=np.float32), float(fps)

def process_video(video_path, roi, accelerated, progress_cb, frame_cb, cancel_flag, seguir_automatico=False):
    intensity, motion, fps = _extract_roi_signals(
        video_path,
        roi,
        accelerated,
        progress_cb,
        frame_cb,
        cancel_flag,
        seguir_automatico=seguir_automatico
    )

    if intensity is None:
        return None

    if len(intensity) < 20:
        raise RuntimeError("No se obtuvieron suficientes frames para analizar el pulso")

    intensity_work, intensity_display = _preprocess_intensity(intensity, fps)
    motion_work, motion_display = _preprocess_motion(motion, fps)

    candidates = [
        {
            "name": "intensity_valleys",
            "work": -intensity_work,
            "display": intensity_display,
        },
        {
            "name": "intensity_peaks",
            "work": intensity_work,
            "display": intensity_display,
        },
        {
            "name": "motion_peaks",
            "work": motion_work,
            "display": motion_display,
        },
    ]

    best_global = {
        "score": -np.inf,
        "bpm": None,
        "peaks": None,
        "display": None,
        "name": None,
        "metadata": {},
    }

    for cand in candidates:
        ref_lag, acf_strength = _estimate_period_acf(
            cand["work"],
            fps,
            bpm_min=BPM_MIN,
            bpm_max=BPM_MAX,
        )

        peaks, bpm, score, metrics = _detect_from_work_signal(
            cand["work"],
            fps,
            ref_lag,
            bpm_min=BPM_MIN,
            bpm_max=BPM_MAX,
        )

        score += 0.20 * acf_strength

        metadata = dict(metrics)
        metadata.update({
            "signal_source": cand["name"],
            "acf_lag_frames": None if ref_lag is None else int(ref_lag),
            "acf_strength": float(acf_strength),
            "score": float(score),
            "bpm_min": float(BPM_MIN),
            "bpm_max": float(BPM_MAX),
            "hit_bpm_ceiling": bool(bpm is not None and bpm >= 0.98 * BPM_MAX),
        })

        if score > best_global["score"] and peaks is not None and bpm is not None:
            best_global["score"] = score
            best_global["bpm"] = bpm
            best_global["peaks"] = peaks
            best_global["display"] = cand["display"]
            best_global["name"] = cand["name"]
            best_global["metadata"] = metadata

    if best_global["peaks"] is None or best_global["bpm"] is None:
        raise RuntimeError("No se detectaron pulsos de forma confiable")

    return (
        float(best_global["bpm"]),
        best_global["display"],
        best_global["peaks"],
        fps,
        best_global["metadata"],
    )