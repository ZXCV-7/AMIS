# Adaptive Multi-scale Framework for Incremental Semantic Segmentation 



## Abtract
Class-incremental semantic segmentation (CISS) aims to progressively learn new categories or adapt to novel environments for semantic segmentation tasks without the need for full model retraining, while ensuring that the segmentation performance on previously learned classes is preserved. Most existing class-incremental semantic segmentation methods mitigate the issues of catastrophic forgetting and background shift through strategies such as pseudo-labeling and knowledge distillation. Although these approaches have achieved certain levels of success, they still exhibit several limitations and challenges：1)Most existing methods overlook the contextual information between tasks, resulting in a fragmented learning process that lacks holistic integration；2) While the knowledge from the old model is directly transferred to the new model, it is important to note that not all of this prior knowledge is necessarily beneficial. To address these challenges, this paper proposes a novel Adaptive Multi-scale Incremental Semantic Segmentation (AMIS) framework incorporating a Global Attention Block (GAB) designed to capture contextual information across different tasks. Furthermore, this paper introduces an Adaptive Multi-scale Distillation Module (AMD) and a Background Compensation Strategy (BCS). The model adaptively focuses on beneficial representations by performing multi-scale pooling and fusion on the features extracted from the decoder. Additionally, the background compensation strategy enhances the model's ability to distinguish ambiguous boundaries between background and target classes. Extensive experiments on the Pascal VOC and ADE20K datasets demonstrate that the proposed method effectively alleviates catastrophic forgetting and background shift issues, outperforming state-of-the-art methods and achieving the highest performance. 
## Requirements
You need to install the following libraries:
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
```
## Datasets
```
data_root/
    ├── VOCdevkit
    │   └── VOC2012/
    │       ├── Annotations/
    │       ├── ImageSet/
    │       ├── JPEGImages/
    │       └── SegmentationClassAug/
    ├── ADEChallengeData2016
    │   ├── annotations
    │   │   ├── training
    │   │   └── validation
    │   └── images
    │       ├── training
    │       └── validation
```
> ### PASCAL VOC 2012
> Download PASCAL VOC2012 devkit (train/val data)
> ``` bash
> sh download_voc.sh
> ```
> - download_voc.sh
> ```bash
>cd data_root
>
>wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
>tar -xf VOCtrainval_11-May-2012.tar
>wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
>wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
>wget http://cs.jhu.edu/~cxliu/data/list.zip
>rm VOCtrainval_11-May-2012.tar
>
>unzip SegmentationClassAug.zip
>unzip SegmentationClassAug_Visualization.zip
>unzip list.zip
>
>mv SegmentationClassAug ./VOCdevkit/VOC2012/
>mv SegmentationClassAug_Visualization ./VOCdevkit/VOC2012/
>mv list ./VOCdevkit/VOC2012/
>
>rm list.zip
>rm SegmentationClassAug_Visualization.zip
>rm SegmentationClassAug.zip
> ```
> ### ADE20k
> ```
> python download_ade20k.py
> ```

## Training & Test
```
sh main.sh
```
We provide a training script ``main.sh``. Detailed training argumnets are as follows:
```sh
python -m torch.distributed.launch --nproc_per_node={num_gpu} --master_port={port} main.py --config ./configs/voc.yaml --log {your_log_name}
```

## Qualitative Results

<img src="./fig/Qualitative Analysis.png" alt="Qualitative Analysis" style="zoom:80%;" />

## Performance comparison

> ### Performance comparison on Pascal VOC

<img src="./fig/Performance comparison on Pascal VOC.png" alt="Performance comparison on Pascal VOC" style="zoom: 80%;" />

> ### Performance comparison on ADE20K

<img src="./fig/Performance comparison on ADE20K.png" alt="Performance comparison on ADE20K" style="zoom: 80%;" />
