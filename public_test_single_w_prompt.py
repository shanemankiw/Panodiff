from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from dataloaders.rota_dataset import SingleDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import time
import os
import datetime
import torch
import sys


# Configs
test=False
data_root_path = 'datasets/sun360_example/raw_crops'
index = '000020'
prompt = 'ski mountain' # the prompt used in the situation
#mask_image_path = '/data/chenziyu/myprojects/OmniDreamer/assets/90binarymask.png'
num_training = 500
exclude_360 = True
batch_size = 1

# Rotation Supervision
rotation_supervise = False
rotation_loss_lambda = 5e-4
roll_augment = True
roll_schedule = False
padding_augment = True


logger_freq = 1 #num_training // batch_size * 2
learning_rate = 1e-5
resume_path = 'norota_clean.ckpt'

NUM_GPUS = 1
N_acc = 2 # gradient accumulate
max_epochs = 20 * N_acc
sd_locked = True
only_mid_control = False
img_size = 512
log_path = 'logs_2312'
name_head = '231225_public'
current_time = datetime.datetime.now()
expname = name_head + current_time.strftime("%Y-%m-%d-%H:%M:%S")
image_logger_path = os.path.join(log_path, expname)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/norota_inpaint.yaml').cuda()
model.load_state_dict(load_state_dict(resume_path, location='cuda:0'), strict=True)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
model.rotation_supervise = rotation_supervise
model.use_gt_rots = True
model.padding_augment = padding_augment
if rotation_supervise:
    model.rotation_loss_lambda = rotation_loss_lambda

model.roll_augment = roll_augment
model.roll_schedule = roll_schedule
model.deform_augment = False

# Misc
test_dataset = SingleDataset(root=data_root_path, 
                                prompt_path='./prompts/my_sun360_prompts_no360.json',
                                dataset_name='sun360',
                                height=img_size,
                                index=index,
                                Train=False,
                                prompt=prompt,
                                )

test_dataloader = DataLoader(test_dataset, 
                num_workers=1, 
                batch_size=batch_size, 
                shuffle=False)


tb_logger = TensorBoardLogger(
    save_dir=log_path,
    name=expname
)

image_callback = ImageLogger(batch_frequency=logger_freq, 
                             save_dir=image_logger_path,
                             tb_logger=tb_logger)

model.eval()
device = model.device#'cuda:0'
#model.to(device)
with torch.no_grad():
    for b_idx, batch in enumerate(test_dataloader):
        for item in batch:
            if isinstance(batch[item], torch.Tensor):
                batch[item] = batch[item].to(device)
        # outputs = model(batch)
        # using flip to check if the left and right is connected
        image_callback.log_img(model, batch, batch_idx=b_idx, split="test", flip=False)

