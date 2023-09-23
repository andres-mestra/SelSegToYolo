import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def get_sam_model(): 
  sam_checkpoint = "sam_vit_l_0b3195.pth"
  model_type = "vit_l"
  device = "cpu"

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)
  predictor = SamPredictor(sam)      
  return predictor