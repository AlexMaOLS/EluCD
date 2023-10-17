# Elucidating-Classifier-Guided-Diffusion

This is the codebase for [Elucidating The Design Space of Classifier-Guided Diffusion Generation](http://arxiv.org/abs/2105.05233).

This repository contains three main folders, targeting the off-the-shelf classifier guidance for [DDPM](https://github.com/openai/guided-diffusion), [EDM](https://github.com/NVlabs/edm) and [DiT](https://github.com/facebookresearch/DiT) respectively.   

# Download pre-trained models
For all the pre-trained diffusion and classifier models, please place them in the `./DDPM/pretrained_models` folder:
For DDPM diffusion models, the ImageNet128x128 Diffusion model [checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt) is from [DDPM diffusion model](https://github.com/openai/guided-diffusion), 

We use the off-the-shelf [Pytorch ResNet classifier](https://pytorch.org/vision/main/models/resnet.html): [ResNet50](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) and [ResNet101](https://download.pytorch.org/models/resnet101-cd907fc2.pth) classifier for guided-sampling. 

For FID evaluation, use the ImageNet 128x128 [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/128/VIRTUAL_imagenet128_labeled.npz) .

# Off-the-shelf Guidance for DDPM 

Firstly go to folder `./DDPM`, which contains all files for the off-the-shelf classifier guidance for [DDPM diffusion model](https://github.com/openai/guided-diffusion). 
you can directly run `./guided_sample.sh`.
All the model checkpoints are stored in the `./pretrained_models/` folder.

For FID evaluation, use `./pytorch-fid-master/src/evaluation_image.sh`, and replace the filename with the sample folder name you created. 

## Off-the-shelf Classifier guidance

Run `./guided_sample.sh`, to generate sample the off-the-shelf classifier guided sampling

 * off-the-shelf ResNet101 DDPM guided:

```
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_type resnet101 --classifier_scale 1.0 --softplus_beta 3.0 --joint_temperature 1.0 --margin_temperature_discount 0.5 --gamma_factor 0.3 --classifier_use_fp16 True"
```


# Results

This table summarizes our ImageNet results for 250 steps of DDPM guided sampling:

| ImageNet 128x128          | FID  | Precision | Recall |
|------------------|------|-----------|--------|
| Diffusion [Baseline](https://github.com/openai/guided-diffusion)   | 5.91 | 0.70      | 0.65   |
| Diffusion [Finetune classifier Guided](https://github.com/openai/guided-diffusion) | 2.97 | 0.78      | 0.59   |
| Diffusion Classifier-Free  | 2.43 | -      | -   |
| Diffusion ResNet50 Guided (Ours) | 2.36 | 0.77      | 0.60   |
| Diffusion ResNet101 Guided (Ours) | 2.19 | 0.79      | 0.58   |



# To be continued

We will release the code of the off-the-shelf classifier guided sampling for [EDM](https://github.com/NVlabs/edm) and [DiT](https://github.com/facebookresearch/DiT) soon.
