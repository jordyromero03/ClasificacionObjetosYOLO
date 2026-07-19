"""
Parte1B — Script de grabación de evidencia
Abrí OBS (u otro grabador de pantalla) ANTES de ejecutar este script.
Presioná:
  g = cambiar a GPU
  c = cambiar a CPU
  q = salir
"""

import subprocess, uuid, time, os
import cv2
import torch
import psutil
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = 'resultados/video_prueba.mp4'
MODEL_PATH = 'models/yolo12n.pt'

MAC = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) for i in range(0, 48, 8)][::-1])


def nvidia_smi_line():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'], timeout=2).decode().strip()
        util, mem_u, mem_t, temp = [x.strip() for x in out.split(',')]
        return f'GPU util:{util}%  VRAM:{mem_u}/{mem_t}MiB  Temp:{temp}C'
    except Exception:
        return 'nvidia-smi N/A'


def draw_overlay(frame, device_label, fps):
    color = (0, 255, 0) if device_label == 'GPU' else (0, 0, 255)
    ram_gb = psutil.virtual_memory().used / 1024**3
    lines = [
        f'{device_label} | FPS: {fps:.1f}',
        f'RAM: {ram_gb:.2f} GB',
        nvidia_smi_line(),
        f'MAC: {MAC}',
    ]
    y = 32
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        y += 26


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'Descargando {MODEL_PATH}...')
        _ = YOLO('yolo12n.pt')
        import shutil; shutil.move('yolo12n.pt', MODEL_PATH)

    model = YOLO(MODEL_PATH)
    source = int(VIDEO_PATH) if str(VIDEO_PATH).isdigit() else VIDEO_PATH

    use_gpu = True
    device_label = 'GPU'
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'No se pudo abrir: {VIDEO_PATH}')
        print('Intentando con webcam 0...')
        cap = cv2.VideoCapture(0)

    win = 'Practica4-1B — YOLOv12n | [g]=GPU  [c]=CPU  [q]=salir'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    print('Ventana abierta. Grabá tu pantalla con OBS ahora.')
    print('g=GPU  c=CPU  q=salir')

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        device_arg = 0 if use_gpu else 'cpu'
        t0 = time.perf_counter()
        res = model(frame, device=device_arg, conf=0.4, verbose=False)
        fps = 1 / max(time.perf_counter() - t0, 1e-6)
        out = res[0].plot()
        draw_overlay(out, device_label, fps)
        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('g'):
            use_gpu = True
            device_label = 'GPU'
        if key == ord('c'):
            use_gpu = False
            device_label = 'CPU'

    cap.release()
    cv2.destroyAllWindows()
    print('Listo. Subí el video grabado a Google Drive.')


if __name__ == '__main__':
    main()
