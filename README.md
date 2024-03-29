# Elucidating-Classifier-Guided-Diffusion (ICLR2024)

This is the codebase for [**Elucidating The Design Space of Classifier-Guided Diffusion Generation**](https://arxiv.org/abs/2310.11311), accepted by ICLR2024 as poster 

This repository contains three main folders, targeting the off-the-shelf classifier guidance for [DDPM](https://github.com/openai/guided-diffusion), [EDM](https://github.com/NVlabs/edm) and [DiT](https://github.com/facebookresearch/DiT) respectively.   

# off-the-shelf classifier guided DDPM 
## Environment variables and files

```
export PYTHONPATH="${PYTHONPATH}:...{Folder_Path}/EluCD-main/DDPM"
pip install blobfile
pip install mpi4py
```

## Download pre-trained models
For all the pre-trained diffusion, classifier models and reference batch, please place them in the `./DDPM/pretrained_models` folder:
For DDPM diffusion models, the ImageNet128x128 [Diffusion model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt) and [fine-tuned classifier](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_classifier.pt) is from [DDPM diffusion model](https://github.com/openai/guided-diffusion), 

We use the off-the-shelf [Pytorch ResNet classifier](https://pytorch.org/vision/main/models/resnet.html): [ResNet50](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) and [ResNet101](https://download.pytorch.org/models/resnet101-cd907fc2.pth) classifier for guided-sampling. 

For FID evaluation, use the ImageNet 128x128 [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz) .

## Off-the-shelf Guidance for DDPM 

Firstly go to folder `./DDPM`, which contains all files for the off-the-shelf classifier guidance for [DDPM diffusion model](https://github.com/openai/guided-diffusion). 
you can directly run `./DDPM/guided_sample.sh`.
All the model checkpoints are stored in the `./DDPM/pretrained_models/` folder.

For FID evaluation, use `./pytorch-fid-master/src/evaluation_image.sh`, and replace the filename with the sample folder name you created. 

## Off-the-shelf Classifier guided DDPM sampling

Run `./guided_sample.sh`, to generate sample the off-the-shelf classifier guided sampling

 * off-the-shelf ResNet101 DDPM guided:

```
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_type resnet101 --classifier_scale 1.0 --softplus_beta 3.0 --joint_temperature 1.0 --margin_temperature_discount 0.5 --gamma_factor 0.3 --classifier_use_fp16 True"
```

## DDPM Results

This table summarizes our conditional ImageNet128x128 generation results for 250 steps of DDPM guided sampling:

| ImageNet 128x128          | FID  | Precision | Recall |
|------------------|------|-----------|--------|
| [Diffusion Baseline](https://arxiv.org/abs/2105.05233)   | 5.91 | 0.70      | 0.65   |
| [Diffusion Finetune classifier Guided](https://arxiv.org/abs/2105.05233) | 2.97 | 0.78      | 0.59   |
| [Diffusion Classifier-Free](https://arxiv.org/pdf/2207.12598.pdf)  | 2.43 | -      | -   |
| Diffusion ResNet50 Guided (Ours) | **2.36** | 0.77      | 0.60   |
| Diffusion ResNet101 Guided (Ours) | **2.19** | 0.79      | 0.58   |


# off-the-shelf classifier guided EDM 

## Download pre-trained models
For all the pre-trained diffusion, classifier models and reference batch, please place them in the `./EDM/pretrained_models` folder:
For EDM diffusion models, the ImageNet64x64 [Diffusion model](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl) is from [EDM diffusion model](https://github.com/NVlabs/edm), 

We use the off-the-shelf [Pytorch ResNet classifier](https://pytorch.org/vision/main/models/resnet.html): [ResNet50](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) and [ResNet101](https://download.pytorch.org/models/resnet101-cd907fc2.pth) classifier for guided-sampling. 

For FID evaluation, use the ImageNet 64x64 [reference batch](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz).

## Off-the-shelf Guidance for EDM 

Firstly go to folder `./EDM`, which contains all files for the off-the-shelf classifier guidance for [EDM diffusion model](https://github.com/NVlabs/edm). 
you can directly run `./EDM/guided_sample.sh`.
All the model checkpoints are stored in the `./EDM/pretrained_models/` folder.

For FID evaluation, use `./pytorch-fid-master/src/evaluation_image.sh`, and replace the filename with the sample folder name you created; the reference batch uses ImageNet 64x64 [reference batch](https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz). 

## Off-the-shelf Classifier guided EDM sampling

Run `./guided_sample.sh`, to generate sample the off-the-shelf classifier guided sampling

## EDM sampling Results

This table summarizes our conditional ImageNet64x64 generation results for diverse sampling steps of EDM-guided sampling:

| ImageNet 64x64          | Classifier  | FID | Steps |
|------------------|------|-----------|--------|
| EDM baseline      | -             |  2.35     | 36  |
| EDM Res101 guided | Off-the-Shelf | **2.22** | 36   |
| EDM baseline      | -             | 2.54      | 18   |
| EDM Res101 guided | Off-the-Shelf | **2.35** | 18   |
| EDM baseline      | -             | 3.64      | 10   |
| EDM Res101 guided | Off-the-Shelf | **3.38** | 10   |


# To be continued

We will release the code of the off-the-shelf classifier guided sampling for [DiT](https://github.com/facebookresearch/DiT) soon.
