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
- Ejecutar el archivo main.py
   - Cambiar el valor de la variable muestra_dir por la ruta de la carpeta donde se tienen las imágenes a segmentar, esta carpeta deberá tener carpetas con cada una de las categorías de imágenes y sus correspondientes archivos .jpg.
   - Cambiar el valor de la variable category_dir_name por el nombre de la carpeta de animal a segmentar y la variable category_name por el nombre de la categoría a segmentar, esta última siendo una referencia para el modelo.
   - Este archivo creará una ventana de matplotlib dispuesta para la segmentación de las imágenes, en este deberá dar click en el contorno y dentro         del objeto a segmentar con la herramienta SAM, luego se cierra la ventana de matplotlib y esto a su vez creará en la carpeta definida en la            variable muestra_dir los archivos .json correspondientes con las información que require Labelme para transformar estos archivos a formato YOLO.
    - Ejecutra el comando en consola para iniciar SelSegToYolo
      ```shell
      python main.py
      ```
- Transformación de archivos creados .json en formato Labelme a formato YOLO
  - Se ejectuta el siguiente comando en consola cambiarndo la ruta donde están los archivos .json mencionados en el punto anterior, los cuales             deben estar en una carpeta aparte sin ningún otro archivo diferente. Comando a correr en consola:
    ```shell
    labelme2yolo --json_dir/path/to/labelme_json_dir/ --val_size 0.15 --test_size 0.15
    ```
  - En la carpeta de los archivos .json quedarán archivos en formato entendible para el modelo YOLO, a esta carpeta se pasarán todas las imágenes que se utilizaron para la segmentación, quedando así las imágenes y sus respectivos archivos en formato YOLO para entrenar el model.
- Modificación del archivo .yaml
    - Del punto anterior se creará un archivo de extensión .yaml, este debe ser modificado poniendo la ruta de los archivos de entrenamiento y los de validación o testeo que arroja el proceso anterior.
- Entrenamiento del modelo YOLOv8
  - Desde la terminal de comandos se debe ejecutar lo siguiente:
  ```shell
    yolo task=detect mode=train data=dataset.yaml model=yolov8m.pt epochs=100 imgsz=256 batch=32
  ```
    Ver documentación: [labelme2yolo](https://pypi.org/project/labelme2yolo/).
  - Este entrenamiento arrojará análisis de métricas de evaluación del modelo, así como también predicciones hechas en las imágenes de validación o testeo.
## SAM MODELS

https://github.com/facebookresearch/segment-anything#model-checkpoints
