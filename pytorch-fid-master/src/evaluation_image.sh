pkill python

#!/bin/bash

python -m pytorch_fid --num-samples 50000 '../../pretrained_models/VIRTUAL_imagenet128_labeled.npz' '../../save_sample_images/resnet101_c1.5_soft3_jt1.0+mt0.5_mu0.3_50000_Imagefolder'


 

