import sys
sys.path.append("..")

import pickle
import cv2
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


sam_checkpoint = "../models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=16)

sa_1b_path = '/mnt/hdd/datasets/SA-1B-1/'
N = 1000

masks_all = []
cos_sim_all = []
for i in tqdm(range(1, N + 1)):
    image = cv2.imread(f'{sa_1b_path}/sa_{i}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)
    masks_all.append(len(masks))
    cos_sim_all.append(mask_generator.predictor.h_cos_sim.detach().cpu().numpy())

    if i >= N:
        break

# save masks_all and cos_sim_all as pickle files
with open(f"outputs/sa1b_masks_all.pkl", "wb") as f:
    pickle.dump(masks_all, f)

with open(f"outputs/sa1b_cos_sim_all.pkl", "wb") as f:
    pickle.dump(cos_sim_all, f)

