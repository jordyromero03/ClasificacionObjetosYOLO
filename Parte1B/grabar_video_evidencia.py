"""
Grabación automática de evidencia — Parte 1B
Graba 40 seg en GPU + 40 seg en CPU y guarda el MP4.
No necesitás OBS: el video se guarda solo.
"""

import subprocess, uuid, time, os, threading
import cv2
import torch
import psutil
import numpy as np
from ultralytics import YOLO

OUTPUT  = os.path.join(os.path.dirname(__file__), 'resultados', 'evidencia_parte1b_jordy.mp4')
MODEL   = os.path.join(os.path.dirname(__file__), 'models', 'yolo12n.pt')
DURACION_GPU = 40   # segundos grabando con GPU
DURACION_CPU = 40   # segundos grabando con CPU

MAC = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff)
                for i in range(0, 48, 8)][::-1])

# nvidia-smi cacheado en hilo aparte para no bloquear el loop de video
_smi_cache = ''
_smi_lock  = threading.Lock()

def _smi_worker():
    global _smi_cache
    while True:
        try:
            out = subprocess.check_output(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                timeout=2).decode().strip()
            u, mu, mt, t = [x.strip() for x in out.split(',')]
            with _smi_lock:
                _smi_cache = f'GPU util:{u}%  VRAM:{mu}/{mt}MiB  Temp:{t}C'
        except Exception:
            with _smi_lock:
                _smi_cache = 'nvidia-smi N/A'
        time.sleep(1)

threading.Thread(target=_smi_worker, daemon=True).start()
time.sleep(0.5)   # esperar primera lectura


def draw_overlay(frame, device_label, fps, segundos_restantes):
    color = (0, 220, 0) if device_label == 'GPU' else (0, 80, 255)
    ram   = psutil.virtual_memory().used / 1024**3
    with _smi_lock:
        smi = _smi_cache
    lines = [
        f'{device_label} | FPS: {fps:.1f}   Tiempo restante: {segundos_restantes:.0f}s',
        f'RAM usada: {ram:.2f} GB',
        smi,
        f'MAC Address: {MAC}',
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    pad        = 6
    line_h     = 28

    # Calcular ancho máximo para el rectángulo de fondo
    max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    box_h = len(lines) * line_h + pad * 2
    box_w = max_w + pad * 2

    # Rectángulo semitransparente negro
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Texto blanco + color del dispositivo en la primera línea
    y = 5 + pad + line_h - 8
    for i, line in enumerate(lines):
        c = color if i == 0 else (230, 230, 230)
        cv2.putText(frame, line, (5 + pad, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (5 + pad, y), font, font_scale, c,        thickness,     cv2.LINE_AA)
        y += line_h


def grabar_segmento(cap, writer, model, device_arg, device_label, duracion, win):
    print(f'\n[{device_label}] Grabando {duracion}s...')
    t_inicio = time.perf_counter()
    frames   = 0
    while True:
        elapsed = time.perf_counter() - t_inicio
        if elapsed >= duracion:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t0  = time.perf_counter()
        res = model(frame, device=device_arg, conf=0.4, verbose=False)
        fps = 1 / max(time.perf_counter() - t0, 1e-6)

        out = res[0].plot()
        draw_overlay(out, device_label, fps, duracion - elapsed)
        writer.write(out)
        cv2.imshow(win, out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Grabación cancelada por el usuario.')
            return False
        frames += 1
        if frames % 30 == 0:
            print(f'  {device_label} | {elapsed:.0f}/{duracion}s | {fps:.1f} FPS')
    return True


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    if not os.path.exists(MODEL):
        print(f'Descargando {MODEL}...')
        import shutil
        _ = YOLO('yolo12n.pt')
        if os.path.exists('yolo12n.pt'):
            shutil.move('yolo12n.pt', MODEL)

    print(f'Modelo:  {MODEL}')
    print(f'MAC:     {MAC}')
    print(f'GPU:     {torch.cuda.get_device_name(0)}')
    print(f'Salida:  {OUTPUT}')
    print(f'\nPonte frente a la camara — la grabacion empieza en 3 segundos...')
    time.sleep(3)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: no se pudo abrir la webcam.')
        return

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f'Webcam:  {w}x{h} @ {fps_src:.0f} FPS')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT, fourcc, fps_src, (w, h))

    model  = YOLO(MODEL)
    win    = 'Parte1B — Evidencia | [q] cancelar'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    ok = grabar_segmento(cap, writer, model, 0,     'GPU', DURACION_GPU, win)
    if ok:
        grabar_segmento(cap, writer, model, 'cpu', 'CPU', DURACION_CPU, win)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    size_mb = os.path.getsize(OUTPUT) / 1024**2
    print(f'\nVideo guardado: {OUTPUT}  ({size_mb:.1f} MB)')
    print('Subi el archivo a Google Drive y pasame el link.')


if __name__ == '__main__':
    main()
