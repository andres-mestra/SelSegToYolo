# SelSegToYolo 🧰 🦾

[[`Paper`](https://github.com/andres-mestra/SelSegToYolo/blob/main/paper/integracion_de_modelos_de_IA_DiveAI.pdf)]

SelSegToYolo es una herramienta que te ayuda con el entrenamiento de un modelo de inteligencia artificial para la clasificación, detección y/o segmentación de imágenes. Usando el poder de segmentación de [Segment Anything](https://segment-anything.com/),la detección y clasificación de [Yolov8](https://yolov8.com/), con este método de entrenamiento, eliminas el "ruido" de las imágenes para que el modelo se centre solo en el objeto de interés, lo que hace que su aprendizaje sea óptimo.

## Instalaciones requeridas

- [Yolov8](https://github.com/ultralytics/ultralytics)
- [FastSam](https://github.com/CASIA-IVA-Lab/FastSAM)

Adicionalmente instalar los cli:

- [labelme2yolo](https://pypi.org/project/labelme2yolo/)
  ```shell
  pip install labelme2yolo
  ```
- [yolo ultralytics](https://github.com/ultralytics/ultralytics)
  ```shell
  pip install ultralytics
  ```

# Ejecución SelSegToYolo

- Clonar el repositorio
  ```shell
  git clone https://github.com/andres-mestra/SelSegToYolo.git
  ```
- Ingresar al directorio de SelSegToYolo, crear una carpeta llamada, **libraries**, luego clonar el repositorio de FastSAM dentro de **libraries**
  ```shell
  cd SelSegToYolo
  mkdir libraries && cd libraries
  git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
  cd FastSAM
  pip install -r requirements.txt
  ```
- Descargar el modelo de SAM que desees utilizar y el modelo de FastSAM, estos deben ir en la raiz del proyecto SelSegToYolo:
  - [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints)
  - [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints)

## SAM MODELS

https://github.com/facebookresearch/segment-anything#model-checkpoints
