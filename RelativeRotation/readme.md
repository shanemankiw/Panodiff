# Predict Relative Rotations

This directory contains the code for predicting the relative rotation between a pair of NFoV images. We provide a toy example for instruction.

**Important**: Our relative rotation models are specifically trained on the SUN360 dataset, and their performance has been validated on both the SUN360 and Laval Indoor datasets. While the model may not ensure accurate results on datasets outside this scope, feel free to experiment if your data shares similarities with SUN360 or Laval Indoor.

## Pretrained Models

Download the pretrained checkpoints for relative rotation prediction modules from [link](https://sjtueducn-my.sharepoint.com/:u:/g/personal/shanemankiw_sjtu_edu_cn/EWCaUyWKFv5NgIKydmITZeEBziCCfW4TdiMWr1tgY78TBQ), and unzip the files into the `ckpts` directory.

## Data Preparation

**Toy Example:**

1. Download the preprocessed small batch test data of SUN360 from [link](https://sjtueducn-my.sharepoint.com/:u:/g/personal/shanemankiw_sjtu_edu_cn/EWCaUyWKFv5NgIKydmITZeEBziCCfW4TdiMWr1tgY78TBQ), and unzip it into `data/sun360_example`
2. `cd data` and run the following command to generate metadata. The file `sun360_example.npy` will be generated under `data`

``` shell
crop_num=50 # number of NFoV images for each panorama
pairs_num=100 # number of generated NFoV pairs for each panorama
raw_data_path=sun360_example

python build_dataset.py $crop_num $pairs_num $raw_data_path
```
3. Replace `data.path` and `data.pairs_file` in `config.yaml` with the corresponding image root and `.npy` file path.

## Relative Pose Prediction
Run the following command to obtain the predicted relative rotations of the input NFoV pairs:

``` shell
GPU=... # Define your GPU Device
save_path=data/pred_results.npy # Define the path for saved predictions

python pred_rotations.py \
    config.yaml \
    --classification_model_path=ckpts/stage1_classification_ckpt.pt \
    --overlap_regression_model_path=ckpts/stage2_overlap_ckpt.pt \
    --nonoverlap_regression_model_path=ckpts/stage2_nonoverlap_ckpt.pt \
    --gpu=$GPU \
    --save_path=$save_path
```

## Acknowledgment
This part of code is heavily based on Cai et al.'s work, [Extreme Rotation Estimation using Dense Correlation Volumes.](https://github.com/RuojinCai/ExtremeRotation_code) Thanks to authors for their great work.
