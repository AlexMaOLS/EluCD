pkill python

#!/bin/bash

# 128*128
SAMPLE_FLAGS="--log_dir ./save_sample_images/resnet101_c1.5_soft3_jt1.0+mt0.5_mu0.3_50000 --save_figure False --batch_size 16 --num_samples 50000 --timestep_respacing 250"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --image_size 128 --learn_sigma True --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True --classifier_scale 1.0 --softplus_beta 3.0 --joint_temperature 1.0 --margin_temperature_discount 0.5 --good_class_factor 1.5 --classifier_use_fp16 True"


mpiexec -n 8 python scripts/classifier_sample.py \  
	--model_path pretrained_models/128x128_diffusion.pt \ 
	--classifier_path pretrained_models/128x128_classifier.pt \
	$MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS


