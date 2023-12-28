data_root = /data/chenziyu/myprojects/PanoData
sun360_root=${data_toor}/my_sun360
pano_num=1000
crop_num=50
pairs_num=50
crop_img_root=${data_toor}/rotation_dataset/pano${pano_num}_pairs${pairs_num}/raw_crops/
crop_meta_path=${data_toor}/rotation_dataset/pano${pano_num}_pairs${pairs_num}/raw_crops/undist/meta.json
raw_data_path=${data_toor}/rotation_dataset/pano${pano_num}_pairs${pairs_num}/raw_crops/undist
meta_out_dir=${data_toor}/rotation_dataset/pano${pano_num}_pairs${pairs_num}/metadata/undist


python scripts/dataset/crop_generator.py $sun360_root $pano_num $crop_num $crop_img_root

python scripts/dataset/build_dataset.py $crop_meta_path $raw_data_path $crop_num $pairs_num $meta_out_dir