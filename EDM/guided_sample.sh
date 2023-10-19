pkill python

#!/bin/bash

torchrun --standalone --nproc_per_node=8 generate.py --outdir='./samples/' --seeds=0-49999 --subdirs --batch=64 --network='./pretrained_models/edm-imagenet-64x64-cond-adm.pkl' --use_classifier 1 --softplus_beta 5.0 --joint_temperature 1.0 --uncond_temperature 0.0 --max_guidance 0.004 --add_factor 0.3 --step 10
