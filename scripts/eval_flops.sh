#!/bin/bash -e

set -x
set -e


num_gpus=1
output_dir="./output9/APE/eval_flops/"
mkdir -p ${output_dir}

timestamp="`date +'%Y%m%d_%H%M%S'`"
LOG=${output_dir}/${timestamp}_log.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


# REC R50
config_files=(
	"configs/REFCOCO_VisualGrounding/ape_deta/ape_deta_r50_12ep.py" # bs=16 for training
	"configs/REFCOCO_VisualGrounding/ape_deta/ape_deta_r50_vlf_12ep.py" # bs=16 for training
)

kwargs="dataloader.train.total_batch_size=8 model.model_vision.test_mask_on=False model.model_vision.test_score_thresh=0.5 model.model_language.max_batch_size=128 model.model_vision.transformer.num_feature_levels=5 "

for config_file in ${config_files[@]}
do
	echo "=============================================================================================="
	echo ${config_file}
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs} model.model_vision.num_classes=1 model.model_vision.select_box_nums_for_evaluation=1 model.model_vision.test_score_thresh=0.5
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs} model.model_vision.num_classes=128 model.model_vision.select_box_nums_for_evaluation=128 model.model_vision.test_score_thresh=0.5
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs} model.model_vision.num_classes=1280 model.model_vision.select_box_nums_for_evaluation=1280 model.model_vision.test_score_thresh=0.5
done


# REC ViT-L
config_files=(
	"configs/REFCOCO_VisualGrounding/ape_deta/ape_deta_vitl_eva02_clip_lsj1024_12ep.py" # bs=8 for training
	"configs/REFCOCO_VisualGrounding/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_12ep.py"  # bs=8 for training
)

kwargs="dataloader.train.total_batch_size=8 model.model_vision.test_mask_on=False model.model_vision.test_score_thresh=0.5 model.model_language.max_batch_size=128 model.model_vision.neck.in_features=[\"p3\",\"p4\",\"p5\",\"p6\"] model.model_vision.mask_in_features=[\"p3\"] model.model_vision.neck.num_outs=5 model.model_vision.transformer.num_feature_levels=5 model.model_vision.backbone.scale_factors=[2.0,1.0,0.5] "
for config_file in ${config_files[@]}
do
	echo "=============================================================================================="
	echo ${config_file}
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs} model.model_vision.num_classes=1 model.model_vision.select_box_nums_for_evaluation=1
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs} model.model_vision.num_classes=128 model.model_vision.select_box_nums_for_evaluation=128
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs} model.model_vision.num_classes=1280 model.model_vision.select_box_nums_for_evaluation=1280
done


# OVD R50
config_files=(
	"configs/COCO_InstanceSegmentation/ape_deta/ape_deta_r50_12ep.py" # bs=16 for training
	"configs/LVIS_InstanceSegmentation/ape_deta/ape_deta_r50_24ep.py" # bs=16 for training
	"configs/COCO_InstanceSegmentation/ape_deta/ape_deta_r50_vlf_12ep.py" # bs=16 for training
	"configs/LVIS_InstanceSegmentation/ape_deta/ape_deta_r50_vlf_24ep.py" # bs=16 for training
)

kwargs="dataloader.train.total_batch_size=8 model.model_vision.test_mask_on=False model.model_vision.test_score_thresh=0.5 model.model_language.max_batch_size=128 model.model_vision.transformer.num_feature_levels=5 "

for config_file in ${config_files[@]}
do
	echo "=============================================================================================="
	echo ${config_file}
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs}
done

# OVD ViT-L
config_files=(
	"configs/COCO_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_clip_lsj1024_cp_12ep.py" # bs=8 for training
	"configs/LVIS_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_clip_lsj1024_cp_24ep.py" # bs=8 for training
	"configs/COCO_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_12ep.py" # bs=8 for training
	"configs/LVIS_InstanceSegmentation/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_24ep.py" # bs=8 for training
)

kwargs="dataloader.train.total_batch_size=8 model.model_vision.test_mask_on=False model.model_vision.test_score_thresh=0.5 model.model_language.max_batch_size=128 model.model_vision.neck.in_features=[\"p3\",\"p4\",\"p5\",\"p6\"] model.model_vision.mask_in_features=[\"p3\"] model.model_vision.neck.num_outs=5 model.model_vision.transformer.num_feature_levels=5 model.model_vision.backbone.scale_factors=[2.0,1.0,0.5] "

for config_file in ${config_files[@]}
do
	echo "=============================================================================================="
	echo ${config_file}
	python3.9 ../detectron2/tools/analyze_model.py --dist-url=tcp://127.0.0.1:49193 --config-file ${config_file} --num-gpus ${num_gpus} --tasks flop -n 1 train.output_dir=${output_dir}/${config_file}/"`date +'%Y%m%d_%H%M%S'`" ${kwargs}
done
