import sys

sys.path.append("..")

from libraries.FastSAM.fastsam import FastSAM, FastSAMPrompt

model = FastSAM('./FastSAM-x.pt')

def get_fast_sam_model(img_path): 
  device = "cpu"
  everything_results = model(img_path, device=device, retina_masks=True, imgsz=256, conf=0.8, iou=0.9,)  
  prompt_process = FastSAMPrompt(img_path, everything_results, device=device)
  return prompt_process