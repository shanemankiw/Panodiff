import torch
import torch.nn.functional as F
import cv2
import numpy as np

img_path = 'datasets/rotation_blip_dataset_train/raw_crops/undist/00565/panorama.jpg'

img = np.asarray(cv2.imread(img_path), dtype=np.float32)/255.0
gt_tensor = torch.from_numpy(img[None, :]).permute([0, 3, 1, 2])

# lateral movements
y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, img.shape[0]), 
                torch.linspace(-1, 1, img.shape[1]))

y_grid = y_grid[None,:]
x_grid = x_grid[None,:]
x_grid = x_grid -0.7
x_grid[x_grid>1] -= 2
x_grid[x_grid<-1] += 2

grid = torch.stack([x_grid, y_grid], dim=3)

sampled_wcorners = F.grid_sample(gt_tensor, grid, align_corners=True).permute([0,2,3,1]).squeeze().numpy()
sampled_wocorners = F.grid_sample(gt_tensor, grid, align_corners=False).permute([0,2,3,1]).squeeze().numpy()

cv2.imwrite('wcorners.png', (sampled_wcorners*255).astype(np.uint8))
cv2.imwrite('wocorners.png', (sampled_wocorners*255).astype(np.uint8))


