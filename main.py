
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
from json import  dump

from get_fast_sam import get_fast_sam_model
from sam_model import get_sam_model
from get_images import get_images_paths
from show_sam_predicts import show_mask, show_points

SAM = get_sam_model()

muestra_dir = 'testeo'
category_dir_name= 'birds' #Carpeta de las imagenes de la categoria
category_dir = muestra_dir + '/' + category_dir_name + '/'
category_name = 'bird' #Categoria solo cambiar nombre
images = get_images_paths(category_dir)
#images = ['birds_26675.jpg','birds_25876.jpg']

points = []
labels = []
def onclick(event):
    global points
    points.append([int(event.xdata), int(event.ydata)])
    labels.append(1)

counter = 1
for imagePath in images:
  #Validate image file
  if imagePath.split('.')[-1] != 'jpg':
    print(f"{imagePath} is not a jpg file")
    continue

  #Reset points
  points = []
  labels = []
  #Load image
  image_dir = category_dir + imagePath
  image = cv2.imread(image_dir)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #show image
  fig = plt.figure(figsize=(10,10))
  plt.imshow(image)
  cid = fig.canvas.mpl_connect('button_press_event', onclick)
  plt.axis('off')
  plt.show()


  #Fast SAM prediction
  input_point = np.array(points)
  input_label = np.array(labels)
  FastSAM = get_fast_sam_model(image_dir)
  masks = FastSAM.point_prompt(points=points, pointlabel=labels)


  plt.figure(figsize=(10,10))
  plt.imshow(image)
  show_mask(masks, plt.gca())
  show_points(input_point, input_label, plt.gca())
  plt.title(f"Img: {imagePath}", fontsize=18)
  plt.axis('off')
  plt.show()

  #Coordinates 
  mask_best = masks[0]
  row, column = np.where(mask_best)
  coordinate_list = np.array([list([row, column]) for row, column in zip(row, column)]).tolist()
  
  is_mask_empty = len(coordinate_list) == 0
  if is_mask_empty:
    print(f"{counter}: {imagePath} 🤦‍♂️")
  else:
    print(f"{counter}: {imagePath}")
    
  #Counter
  counter += 1
  

  #Image to base64
  _, imageBuffer = cv2.imencode('.jpg', image)
  image_encoded = base64.b64encode(imageBuffer).decode('utf-8')

  #Image info
  image_info = {
    "version": "3.16.7",
    "flags":  {},
    "shapes": [
      {  
        "label": category_name,
        "line_color": None,
        "fill_color": None,
        "shape_type": "polygon",
        "flags":  {},
        "points": coordinate_list
      }
    ],
    "lineColor": [ 0, 255, 0,128],
    "fillColor": [255,0,0,128],
    "imagePath": imagePath,
    "imageData": image_encoded
  }

  image_json_dir = image_dir.replace(".jpg", ".json")
 
  with open(image_json_dir, "w") as archivo:
    dump(image_info, archivo, indent=2)