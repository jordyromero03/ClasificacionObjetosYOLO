# Práctica 4-1: Visión por Computador con GPU

**Universidad Politécnica Salesiana — Visión por Computador**  
**Período Lectivo:** Marzo – Agosto 2026  
**Docente:** Ing. Vladimir Robles Bykbaev  
**Estudiante:** Jordy Romero

---

## Estructura del proyecto

```
Practica4-1/
├── Parte1A/                          # YOLOv11 — Segmentación de señales de tránsito
│   ├── Parte1A_YOLOv11_Segmentacion.ipynb
│   ├── dataset_traffic/              # Dataset en formato YOLO
│   │   ├── data.yaml
│   │   ├── train/images/ + labels/
│   │   ├── valid/images/ + labels/
│   │   └── test/images/  + labels/
│   ├── models/
│   │   ├── yolo11n-seg.pt            # Modelo base preentrenado
│   │   └── best.pt                   # Modelo entrenado (generado al ejecutar)
│   ├── resultados/                   # Imágenes de salida generadas
│   └── runs/                         # Logs y pesos de entrenamiento (YOLO)
│
├── Parte1B/                          # YOLOv12 + Real-ESRGAN — GPU vs CPU
│   ├── Parte1B_YOLOv12_SuperResolucion.ipynb
│   ├── models/
│   │   ├── yolo12n.pt
│   │   └── RealESRGAN_x4plus.pth
│   └── resultados/
│       └── video_prueba.mp4          # Video de entrada para el benchmark
│
└── Parte1C/                          # Pipeline OpenCV CUDA — C++
    ├── main.cpp
    ├── CMakeLists.txt
    ├── build/practica4_1c            # Binario compilado
    └── resultados/
```

---

## Requisitos

- Python 3.10+, PyTorch con CUDA, Ultralytics, OpenCV, spandrel, psutil
- CUDA Toolkit (probado con 13.3), GPU NVIDIA
- Para Parte1C: CMake, OpenCV5 con módulos CUDA

---

## Ejecución

### Parte 1A — Segmentación YOLOv11
Abrir y ejecutar celda por celda en VS Code o JupyterLab:
```
Parte1A/Parte1A_YOLOv11_Segmentacion.ipynb
```

### Parte 1B — Benchmark YOLOv12 + Real-ESRGAN
```
Parte1B/Parte1B_YOLOv12_SuperResolucion.ipynb
```

### Parte 1C — Pipeline C++ con CUDA
```bash
cd Parte1C
./build/practica4_1c imagen.jpg    # con imagen
./build/practica4_1c 0             # con webcam
```

---

## Cambiar el dataset (Parte 1A)

El notebook de Parte1A está diseñado para que sea fácil sustituir el dataset sin modificar el código principal. Solo se tocan dos lugares.

### Opción A — Dataset ya en formato YOLO (recomendado)

Si descargaste un dataset de Roboflow, Kaggle u otra fuente que ya incluye carpetas `train/`, `valid/`, `test/` con subcarpetas `images/` y `labels/`, sigue estos pasos:

1. Copia o mueve el dataset a `Parte1A/` con el nombre que prefieras, por ejemplo `dataset_nuevo/`.

2. Verifica que la estructura interna sea:
   ```
   dataset_nuevo/
   ├── data.yaml
   ├── train/
   │   ├── images/   (archivos .jpg o .png)
   │   └── labels/   (archivos .txt con anotaciones YOLO)
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

3. Abre el notebook y modifica la **celda 2** (bloque de configuración):
   ```python
   # Antes:
   DATASET = 'dataset_traffic'
   yaml_path = os.path.join(DATASET, 'data.yaml')

   # Después:
   DATASET = 'dataset_nuevo'
   yaml_path = os.path.join(DATASET, 'data.yaml')
   ```

4. Edita `dataset_nuevo/data.yaml` para que el campo `path` apunte a la ubicación absoluta correcta:
   ```yaml
   path: /home/TU_USUARIO/Documents/.../Parte1A/dataset_nuevo
   train: train/images
   val:   valid/images
   test:  test/images
   nc: 3
   names: ['clase0', 'clase1', 'clase2']
   ```
   Cambia `nc` y `names` según las clases de tu nuevo dataset.

5. Si el número de clases cambia, elimina `models/best.pt` para que el entrenamiento parta desde el modelo base `yolo11n-seg.pt`.

6. Ejecuta el notebook desde la celda 1. El resto funciona automáticamente.

---

### Opción B — Dataset con anotaciones en otro formato (COCO, Pascal VOC, etc.)

Si tu dataset usa anotaciones JSON (formato COCO) o XML (Pascal VOC), necesitas convertirlas al formato YOLO antes de continuar.

**Conversión desde COCO JSON:**
```bash
pip install ultralytics
yolo data convert --format coco --source dataset_coco/annotations/instances_train.json --output dataset_nuevo/
```

O manualmente: cada imagen necesita un archivo `.txt` con una línea por objeto:
```
class_id x_center y_center width height
```
donde todos los valores están normalizados entre 0 y 1.

Para segmentación (como en este proyecto), el formato es poligonal:
```
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```
con las coordenadas de cada vértice del polígono normalizadas.

---

### Formato de etiquetas YOLO segmentación (referencia)

Cada archivo `.txt` en `labels/` corresponde a una imagen y contiene una línea por objeto detectado:

```
2 0.512 0.310 0.538 0.290 0.561 0.312 0.545 0.380 0.502 0.375
```

- El primer número es el `class_id` (entero, empezando en 0).
- Los pares siguientes son las coordenadas `x y` de los vértices del polígono.
- Todos los valores están normalizados: divididos por el ancho/alto de la imagen.
- El archivo `.txt` tiene el mismo nombre que la imagen (cambiando la extensión).

---

### Resumen rapido

| Que cambiar         | Donde                                        |
|---------------------|----------------------------------------------|
| Nombre del dataset  | Variable `DATASET` en celda 2 del notebook   |
| Ruta absoluta       | Campo `path:` en `data.yaml`                 |
| Numero de clases    | Campos `nc:` y `names:` en `data.yaml`       |
| Modelo base         | `models/yolo11n-seg.pt` (no tocar si nc = 3) |
| Borrar modelo viejo | `models/best.pt` (si cambias las clases)     |
