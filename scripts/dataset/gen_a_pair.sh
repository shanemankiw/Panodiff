sun360_root=/HDD/22Ubuntu/nfs_share/sun360/full
pano_num=50
crop_num=50
crop_img_root=datasets/a_pair/raw_crops/
crop_meta_path=datasets/a_pair/raw_crops/undist/meta.json
raw_data_path=datasets/a_pair/raw_crops/undist
meta_out_dir=datasets/a_pair/metadata/my_sun360/undist/
pairs_num=20

python scripts/dataset/crop_generator.py $sun360_root $pano_num $crop_num $crop_img_root

python scripts/dataset/build_dataset.py $crop_meta_path $raw_data_path $crop_num $pairs_num $meta_out_dir