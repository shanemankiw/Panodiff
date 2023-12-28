from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from dataloaders.rota_dataset import RotationDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import time
import os
import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import sys

use_rota_pred_rots = True
data_root_path = 'datasets/sun360_example/raw_crops'
pair_path = 'datasets/sun360_example/meta/sun360_example.npy'
resume_path = 'pretrained_models/processed/rota_control_sd.ckpt'

stage_name="tune-ctrlnet"
num_training = 10000
batch_size = 1
exclude_360 = True
logger_freq = 50
every_n_train_steps= 1250 


###
### rotation train and trick
###
use_gt_rots = True
rotation_supervise = False

# equ-varient loss
use_equivarient_loss = False
equi_loss_lambda = 0.0

# augmentation tricks
roll_augment = True
deform_augment = False
padding_augment = False

# deprecated; just keep it false
use_pred_rots = False


'''
TODO (Done) 
Start a new train script.
'''

NUM_GPUS = 1
N_acc = 1 # gradient accumulate
max_epochs = 5 * N_acc
sd_locked = True
only_mid_control = False
img_size = 128
log_path = 'logs'
name_head = 'training'

current_time = datetime.datetime.now()
expname = name_head + current_time.strftime("%Y-%m-%d-%H-%M-%S")
image_logger_path = os.path.join(log_path, expname)
ckpt_logger_path = os.path.join(log_path, expname, 'ckpts')


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/norota_inpaint.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
model.down_scale = 2

# rotations
model.use_gt_rots = use_gt_rots

#invarient loss
model.use_equivarient_loss = use_equivarient_loss
if use_equivarient_loss:
    model.equi_loss_lambda = equi_loss_lambda

# roll augment
model.roll_augment = roll_augment
model.deform_augment = deform_augment
model.padding_augment = padding_augment

model.use_pred_rots = use_pred_rots

model.learning_rate = 1e-5
print("training controlnet, fix rotanet")
print("set lr of controlnet to: ", model.learning_rate)


# Misc
train_dataset = RotationDataset(root=data_root_path,
                                num_images=num_training,
                                pairs_file=pair_path,
                                height=img_size,
                                Train=True,
                                down_scale=2)

train_dataloader = DataLoader(train_dataset,
                num_workers=1, # 16
                batch_size=batch_size,
                shuffle=True)


tb_logger = TensorBoardLogger(
    save_dir=log_path,
    name=expname
)

image_callback = ImageLogger(batch_frequency=logger_freq,
                             save_dir=image_logger_path,
                             tb_logger=tb_logger)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="global_step",
    mode='max',
    every_n_train_steps=every_n_train_steps,
    dirpath=ckpt_logger_path,
)

trainer = pl.Trainer(
                    logger=tb_logger,
                    max_epochs=max_epochs,
                    gpus=NUM_GPUS,
                    precision=32,
                    accumulate_grad_batches=N_acc,
                    callbacks=[image_callback, checkpoint_callback],
                    log_every_n_steps=logger_freq, # I just added
                    # strategy="ddp", #uncomment to apply ddp
                    )

# Train!
trainer.fit(model, train_dataloader)
