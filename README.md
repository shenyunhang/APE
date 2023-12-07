# APE: Aligning and Prompting Everything All at Once for Universal Visual Perception

<p align="center">
    <img src="./.asset/ape.png" width="96%" height="96%">
</p>


<font size=7><div align='center' > :grapes: \[[Read our arXiv Paper](https://arxiv.org/abs/2312.02153)\] &nbsp; :apple: \[[Try our Online Demo](https://huggingface.co/spaces/shenyunhang/APE)\] </div></font>

---

<p align="center">
    <img src="./.asset/example_1.png" width="96%" height="96%">
</p>


## :bulb: Highlight

- **High Performance.**  SotA (or competitive) performance on **160** datasets with only one model.
- **Perception in the Wild.** Detect and segment **everything** with thousands of vocabularies or language descriptions all at once.
- **Flexible.** Support both foreground objects and background stuff for instance segmentation and semantic segmentation.


## :label: TODO 

- [x] Release inference code and demo.
- [x] Release checkpoints.
- [x] Release training codes.
- [ ] Add clean docs.


## :hammer_and_wrench: Install 

1. Clone the APE repository from GitHub:

```bash
git clone https://github.com/shenyunhang/APE
cd APE
```

2. Install the required dependencies and APE:

```bash
pip3 install -r requirements.txt
python3 -m pip install -e .
```


## :arrow_forward: Demo Localy

**Web UI demo**
```
pip3 install gradio
cd APE/demo
python3 app.py
```
If you have GPUs, this demo will detect them and use one GPU.

Please feel free to try our [Online Demo](https://huggingface.co/spaces/shenyunhang/APE)!

<p align="center">
<img src="./.asset/demo.png" width="96%" height="96%">
</p>


## :books: Data Prepare
Following [here](https://github.com/shenyunhang/APE/blob/main/datasets/README.md) to prepare the following datasets:

|       |   COCO  |   LVIS  | Objects365 | Openimages | VisualGenome |  SA-1B  | RefCOCO |   GQA   | PhraseCut | Flickr30k |  ODinW  |  SegInW | Roboflow100 |  ADE20k | ADE-full |  BDD10k | Cityscapes |  PC459  |   PC59  |   VOC   |    D3   |
|:-----:|:-------:|:-------:|:----------:|:----------:|:------------:|:-------:|:-------:|:-------:|:---------:|:---------:|:-------:|:-------:|:-----------:|:-------:|:--------:|:-------:|:----------:|:-------:|:-------:|:-------:|:-------:|
| Train | &check; | &check; |   &check;  |   &check;  |    &check;   | &check; | &check; | &check; |  &check;  |  &check;  | &cross; | &cross; |   &cross;   | &cross; |  &cross; | &cross; |   &cross;  | &cross; | &cross; | &cross; | &cross; |
|  Test | &check; | &check; |   &check;  |   &check;  |    &cross;   | &cross; | &check; | &cross; |  &cross;  |  &cross;  | &check; | &check; |   &check;   | &check; |  &check; | &check; |   &check;  | &check; | &check; | &check; | &check; |


## :test_tube: Inference

### Infer on 160+ dataset
We provide several scripts to evaluate all models.

It is necessary to adjust the checkpoint location and GPU number in the scripts before running them.

```bash
scripts/eval_all_D.sh
scripts/eval_all_C.sh
scripts/eval_all_B.sh
scripts/eval_all_A.sh
```

### Infer on images or videos

APE-D
```
python3.9 demo/demo_lazy.py \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py \
--input image1.jpg image2.jpg image3.jpg \
--output /path/to/output/dir \
--confidence-threshold 0.1 \
--text-prompt 'person,car,chess piece of horse head' \
--with-box \
--with-mask \
--with-sseg \
--opts \
train.init_checkpoint=/path/to/APE-D/checkpoint \
model.model_vision.select_box_nums_for_evaluation=500 \
model.model_vision.text_feature_bank_reset=True \
```


## :train: Training

### Prepare backbone and language models
```bash
git lfs install
git clone https://huggingface.co/QuanSun/EVA-CLIP models/QuanSun/EVA-CLIP/
git clone https://huggingface.co/BAAI/EVA models/BAAI/EVA/
git clone https://huggingface.co/Yuxin-CV/EVA-02 models/Yuxin-CV/EVA-02/
```

Resize patch size:
```bash
python3.9 tools/eva_interpolate_patch_14to16.py --input models/QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt --output models/QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14to16_plus_s9B.pt --image_size 224
python3.9 tools/eva_interpolate_patch_14to16.py --input models/QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt --output models/QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14to16_s11B.pt --image_size 224
python3.9 tools/eva_interpolate_patch_14to16.py --input models/QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt --output models/QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14to16_s6B.pt --image_size 336
```

### Train APE-D

Single node:
```bash
python3.9 tools/train_net.py \
--num-gpus 8 \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_`date +'%Y%m%d_%H%M%S'`
```

Multiple nodes:
```bash
python3.9 tools/train_net.py \
--dist-url="tcp://${MASTER_IP}:${MASTER_PORT}" \
--num-gpus ${HOST_GPU_NUM} \
--num-machines ${HOST_NUM} \
--machine-rank ${INDEX} \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_`date +'%Y%m%d_%H'`0000
```

### Train APE-C

Single node:
```bash
python3.9 tools/train_net.py \
--num-gpus 8 \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k_`date +'%Y%m%d_%H%M%S'`
```

Multiple nodes:
```bash
python3.9 tools/train_net.py \
--dist-url="tcp://${MASTER_IP}:${MASTER_PORT}" \
--num-gpus ${HOST_GPU_NUM} \
--num-machines ${HOST_NUM} \
--machine-rank ${INDEX} \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k_`date +'%Y%m%d_%H'`0000
```

### Train APE-B

Single node:
```bash
python3.9 tools/train_net.py \
--num-gpus 8 \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k_`date +'%Y%m%d_%H%M%S'`
```

Multiple nodes:
```bash
python3.9 tools/train_net.py \
--dist-url="tcp://${MASTER_IP}:${MASTER_PORT}" \
--num-gpus ${HOST_GPU_NUM} \
--num-machines ${HOST_NUM} \
--machine-rank ${INDEX} \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k_`date +'%Y%m%d_%H'`0000
```

### Train APE-A

Single node:
```bash
python3.9 tools/train_net.py \
--num-gpus 8 \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj1024_cp_720k.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj1024_cp_720k_`date +'%Y%m%d_%H%M%S'`
```

Multiple nodes:
```bash
python3.9 tools/train_net.py \
--dist-url="tcp://${MASTER_IP}:${MASTER_PORT}" \
--num-gpus ${HOST_GPU_NUM} \
--num-machines ${HOST_NUM} \
--machine-rank ${INDEX} \
--resume \
--config-file configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj1024_cp_720k.py \
train.output_dir=output/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj1024_cp_720k_`date +'%Y%m%d_%H'`0000
```



## :luggage: Checkpoints

```
git lfs install
git clone https://huggingface.co/shenyunhang/APE
```

<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>APE-A</td>
      <td><a href="https://huggingface.co/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj_cp_720k_20230504_002019/model_final.pth">HF link</a></td>
      <td><a href="https://github.com/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj1024_cp_720k.py">link</a></td>
    </tr>
    <tr>
      <th>2</th>
      <td>APE-B</td>
      <td><a href="https://huggingface.co/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_225418/model_final.pth">HF link</a> 
      <td><a href="https://github.com/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py">link</a></td>
    </tr>
    <tr>
      <th>3</th>
      <td>APE-C</td>
      <td><a href="https://huggingface.co/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_210950/model_final.pth">HF link</a> 
      <td><a href="https://github.com/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py">link</a></td>
    </tr>
    <tr>
      <th>4</th>
      <td>APE-D</td>
      <td><a href="https://huggingface.co/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth">HF link</a> 
      <td><a href="https://github.com/shenyunhang/APE/blob/main/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl.py">link</a></td>
    </tr>
  </tbody>
</table>


## :medal_military: Results

<img src=".asset/radar.png" alt="radar" width="100%">


## :black_nib: Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@article{shen2023aligning,
  title={Aligning and Prompting Everything All at Once for Universal Visual Perception},
  author={Yunhang Shen and Chaoyou Fu and Peixian Chen and Mengdan Zhang and Ke Li and Xing Sun and Yunsheng Wu and Shaohui Lin and Rongrong Ji},
  journal={arXiv preprint arXiv:2312.02153},
  year={2023}
}
```
