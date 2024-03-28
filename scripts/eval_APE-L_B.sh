#!/bin/bash -e

set -x
set -e


kwargs=""
init_checkpoint="output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_225418/model_final.pth"

num_gpus=7
output_dir="./output9/APE/eval_APE-L_B/"


config_files=(
	"configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py"
	"configs/COCO_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_12ep.py"
	"configs/COCO_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/ODinW_Detection/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_13.py"
	"configs/ODinW_Detection/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_35.py"
	"configs/SegInW_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/Roboflow_Detection/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/ADE20k_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/ADE20k_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/ADE20kFull_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/BDD10k_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/BDD10k_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/Cityscapes_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/PascalContext459_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/PascalContext59_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/PascalVOC20_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
	"configs/D3_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024.py"
)

for config_file in ${config_files[@]}
do
	echo "=============================================================================================="
	echo ${config_file}
	python3 tools/train_net.py --eval-only --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" train.init_checkpoint=${init_checkpoint} ${kwargs}
done
