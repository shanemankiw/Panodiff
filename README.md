# Panodiff
![image](assets/teaser.png)

## [Paper](https://arxiv.org/abs/2308.14686) |  [Video](https://www.youtube.com/watch?v=CGqEnUzpWWQ&t=14s)

Official implementation of the ACM Multimedia 2023 paper '360-Degree Panorama Generation from Few Unregistered NFoV Images'.



## Rotation Estimation

Please refer to the data preparation part [here](RelativeRotation/readme.md) in 'RelativeRotation/' folder, and prepare the sample dataset.

## Prerequisites
You can follow this to setup your python environment:
```
conda env create -f environment.yaml
conda activate pano
```

## Download Pretrained Models

The pretrained ckpts could be found in this OneDrive [Link](https://sjtueducn-my.sharepoint.com/:u:/g/personal/shanemankiw_sjtu_edu_cn/EWCaUyWKFv5NgIKydmITZeEBziCCfW4TdiMWr1tgY78TBQ):

Please put pretrained_models/ under the main folder. It should be of this file structure:

```
pretrained_models/
  -processed/
    -rota_control_sd.ckpt
  -norota_clean.ckpt
```



## Usage

After generating the datasets, please set the 'data_root_path' and the 'pair_path' in scripts to where you put your generated datasets and generated pair information. For example:

```
data_root_path = 'datasets/sun360_example/raw_crops'
pair_path = 'datasets/sun360_example/meta/sun360_example.npy'
# some additional settings could also be found in each script
```

Then we could 

```
# Test on the complete test set
python public_test_on_sampleset.py 

# Train on the complete train set
python public_train_on_sampleset.py 

# Prompt Editing with pair input.
python public_test_pair_w_prompt.py 
# Prompt Editing with single input. 
python public_test_single_w_prompt.py 
```

Note that the path and additional settings should be adjusted for each python script.



## Acknowledgement

Our code is heavily based on [ControlNet](https://github.com/lllyasviel/ControlNet), thanks to the authors.

We also would like to thank all authors who provided their code for us, including [SIG-SS](https://github.com/hara012/sig-ss), [OmniDreamer](https://github.com/akmtn/OmniDreamer) and [StyleLight](https://github.com/Wanggcong/StyleLight), and huge thanks to the authors of [ImmerseGAN](https://lvsn.github.io/ImmerseGAN/) for helping us run the test results.

## Citation

Cite as below if you find this repository is helpful to your project:

```
@inproceedings{wang2023360,
  title={360-Degree Panorama Generation from Few Unregistered NFoV Images},
  author={Wang, Jionghao and Chen, Ziyu and Ling, Jun and Xie, Rong and Song, Li},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={6811--6821},
  year={2023}
}
```
