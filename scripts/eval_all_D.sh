#!/bin/bash -e

set -x
set -e

kwargs="model.model_vision.transformer.proposal_ambiguous=1"
init_checkpoint="output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4x270k_20230829_162438/model_final.pth"
output_dir="output2/eval_all/D_20230829_162438/"


num_gpus=7


config_files=(
	"configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
	"configs/COCO_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_12ep.py"
	"configs/COCO_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/ODinW_Detection/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_13.py"
	"configs/ODinW_Detection/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_35.py"
	"configs/SegInW_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/Roboflow_Detection/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/ADE20k_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/ADE20k_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/ADE20kFull_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/BDD10k_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/BDD10k_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/Cityscapes_PanopticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/PascalContext459_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/PascalContext59_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/PascalVOC20_SemanticSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
	"configs/D3_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024.py"
)

for config_file in ${config_files[@]}
do
	echo "=============================================================================================="
	echo ${config_file}
	python3.9 tools/train_net.py --eval-only --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" train.init_checkpoint=${init_checkpoint} ${kwargs}
done
