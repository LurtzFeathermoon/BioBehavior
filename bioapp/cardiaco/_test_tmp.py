import sys, os, tempfile
sys.path.insert(0, r"D:\Documents\AbacusAI\BioBehaviour")
import numpy as np
import cv2
from bioapp.cardiaco.procesador import _extract_roi_signals, _preprocess_intensity, _estimate_period_acf, _detect_from_work_signal

W, H = 320, 240
FPS = 30.0
N = 240

def gen_video(path, movimiento):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    cx0, cy0 = 160.0, 120.0
    body_len, body_w = 90, 30
    heart_local = (20, 0)
    pulse_hz = 3.0
    for i in range(N):
        frame = np.full((H, W, 3), 200, dtype=np.uint8)
        t = i / FPS
        if movimiento:
            cx = cx0 + 0.4 * i
            cy = cy0 + 0.1 * i
            angle_deg = 15.0 * t
        else:
            cx, cy, angle_deg = cx0, cy0, 0.0
        # ruido leve para simular una silueta real
        noise = np.random.normal(0, 1.5)
        box = cv2.boxPoints(((cx, cy), (body_len, body_w), angle_deg))
        cv2.fillConvexPoly(frame, box.astype(int), (90, 90, 90))
        cv2.randu(frame, 195, 205) if False else None
        rad = np.radians(angle_deg)
        hx = cx + heart_local[0]*np.cos(rad) - heart_local[1]*np.sin(rad)
        hy = cy + heart_local[0]*np.sin(rad) + heart_local[1]*np.cos(rad)
        pulse = 40*(0.5+0.5*np.sin(2*np.pi*pulse_hz*t))
        color = int(90+pulse)
        cv2.circle(frame, (int(hx), int(hy)), 6, (color,color,color), -1)
        # ruido de cámara
        ruido = np.random.normal(0, 2.0, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16)+ruido, 0, 255).astype(np.uint8)
        vw.write(frame)
    vw.release()
    return (cx0, cy0, heart_local)

for movimiento in [False, True]:
    path = os.path.join(tempfile.gettempdir(), f"test_org_{movimiento}.mp4")
    cx0, cy0, heart_local = gen_video(path, movimiento)
    roi0 = (int(cx0+heart_local[0]-10), int(cy0+heart_local[1]-10), 20, 20)

    angulos = []
    def frame_cb(frame, roi):
        if len(roi) == 5:
            angulos.append(roi[4])

    intensity, motion, fps = _extract_roi_signals(path, roi0, False, None, frame_cb, lambda: False, seguir_automatico=True)
    work, disp = _preprocess_intensity(intensity, fps)
    ref_lag, s = _estimate_period_acf(work, fps)
    peaks, bpm, score, metrics = _detect_from_work_signal(work, fps, ref_lag)
    angulos = np.array(angulos)
    print(f"movimiento={movimiento}  bpm={bpm}  angulo_std={angulos.std():.2f}  angulo_max_abs={np.abs(angulos).max():.2f}  angulo_final={angulos[-1]:.2f}")
