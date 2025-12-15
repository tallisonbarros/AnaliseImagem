import os
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

checkpoint = os.path.join(os.getcwd(), "sam_vit_h_4b8939.pth")
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# exemplo fake: set_image precisaria de um array RGB real
dummy = np.zeros((10, 10, 3), dtype=np.uint8)
predictor.set_image(dummy)
mask, scores, _ = predictor.predict(
    point_coords=np.array([[5, 5]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.int32),
    multimask_output=True
)

print(f"SAM1/Predictor carregado. Mascara teste gerada. Scores: {scores}")
