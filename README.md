# Practica 4-1: Clasificacion de Objetos con Deep Learning y GPU

**Universidad Politecnica Salesiana — Vision por Computador**
**Periodo Lectivo:** Marzo – Agosto 2026
**Docente:** Ing. Vladimir Robles Bykbaev
**Autores:** Michael Lata · Jordy Romero (Grupo 8)

---

## Estructura del proyecto

```
Practica4-1/
├── informe/
│   ├── main.tex                          # Fuente LaTeX del informe unificado
│   ├── RomeroJ_lataM_Practica4.pdf       # Informe entregable
│   └── capturas/                         # Imagenes incluidas en el informe
│
├── Parte1A/                              # YOLOv11n-seg — Segmentacion (equipo)
│   ├── Parte1A_YOLOv11_Segmentacion.ipynb
│   ├── dataset_traffic/                  # Dataset YOLO con 3 clases
│   │   ├── data.yaml
│   │   ├── train/ valid/ test/
│   ├── models/
│   │   ├── best.pt                       # Modelo entrenado (incluido)
│   │   └── best.onnx                     # Exportacion ONNX (incluida)
│   └── resultados/
│
├── Parte1B/                              # YOLOv12n+YOLOv26n / RealPLKSR (individual)
│   ├── Parte1B_YOLOv12_SuperResolucion.ipynb
│   ├── grabar_video_evidencia.py         # Script de grabacion de evidencia
│   ├── models/
│   │   └── 4xNomosWebPhoto_RealPLKSR.pth  # Descargar manualmente (ver abajo)
│   └── resultados/
│
└── Parte1C/                              # Pipeline OpenCV CUDA C++ (individual)
    ├── main.cpp
    ├── benchmark_cpu.cpp
    ├── CMakeLists.txt
    └── build/                            # Generado al compilar (no incluido)
```

---

## Requisitos

- Python 3.10+, PyTorch con CUDA, Ultralytics, OpenCV, spandrel, psutil
- CUDA Toolkit 12+ y GPU NVIDIA
- Para Parte1C: CMake 3.20+, OpenCV5 compilado con modulos CUDA

Instalar dependencias Python:
```bash
pip install ultralytics spandrel psutil opencv-python torch torchvision
```

---

## Configuracion inicial

### Modelos base YOLO (se descargan automaticamente)
Los modelos `yolo11n-seg.pt`, `yolo12n.pt` y `yolo26n.pt` no estan en el repositorio.
Ultralytics los descarga automaticamente al ejecutar los notebooks por primera vez.

### Modelo de super resolucion RealPLKSR (descarga manual)
El archivo `.pth` no esta en el repositorio por su tamaño. Descargarlo desde:

```
https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR
```

Guardarlo en:
```
Parte1B/models/4xNomosWebPhoto_RealPLKSR.pth
```

---

## Ejecucion

### Parte 1A — Segmentacion YOLOv11n-seg (equipo)
```bash
# Abrir y ejecutar en VS Code o JupyterLab
Parte1A/Parte1A_YOLOv11_Segmentacion.ipynb
```
El modelo `best.pt` ya esta entrenado e incluido. Para reentrenar, ejecutar
desde la celda de entrenamiento.

### Parte 1B — Deteccion y Super Resolucion GPU vs CPU (individual)
```bash
# Benchmark completo
Parte1B/Parte1B_YOLOv12_SuperResolucion.ipynb

# Grabacion de video de evidencia (requiere webcam)
cd Parte1B
python3 grabar_video_evidencia.py
```

### Parte 1C — Pipeline OpenCV CUDA C++ (individual)
```bash
cd Parte1C
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Benchmark sobre imagen fija (genera mosaicos en build/resultados/)
./build/practica4_1c imagen.jpg

# Modo webcam en vivo (g=GPU, c=CPU, q=salir)
./build/practica4_1c 0
```

---

## Cambiar el dataset (Parte 1A)

Para sustituir el dataset sin modificar el codigo principal:

1. Copiar el nuevo dataset a `Parte1A/` con estructura YOLO:
   ```
   dataset_nuevo/
   ├── data.yaml
   ├── train/images/  y  train/labels/
   ├── valid/images/  y  valid/labels/
   └── test/images/   y  test/labels/
   ```
   El formato de etiquetas de segmentacion es poligonal:
   ```
   class_id x1 y1 x2 y2 ... xn yn   (coordenadas normalizadas 0-1)
   ```

2. Editar la celda de configuracion del notebook:
   ```python
   DATASET = 'dataset_nuevo'   # antes: 'dataset_traffic'
   ```

3. Editar `dataset_nuevo/data.yaml`:
   ```yaml
   path: /ruta/absoluta/a/Parte1A/dataset_nuevo
   train: train/images
   val:   valid/images
   test:  test/images
   nc: 3                        # numero de clases
   names: ['clase0', 'clase1', 'clase2']
   ```

4. Borrar `models/best.pt` si cambias el numero de clases, para que el
   entrenamiento parta desde el modelo base.

Para datasets en formato COCO o Pascal VOC, convertir primero con:
```bash
# Desde COCO JSON
yolo data convert --format coco \
     --source anotaciones/instances_train.json \
     --output dataset_nuevo/
```

En Roboflow: seleccionar **Task = Instance Segmentation** y descargar
en formato **YOLOv8**.

---

## Informe

El informe unificado (Parte 1A equipo + Parte 1B/1C individual por cada autor)
esta en `informe/RomeroJ_lataM_Practica4.pdf`.

Para recompilar el PDF:
```bash
cd informe
pdflatex main.tex
pdflatex main.tex   # segunda pasada para referencias cruzadas
```
