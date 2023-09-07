import numpy as np
import random
from PIL import Image
import os


num_sample = 50000
data_path = './save_sample_images/resnet101_c1.5_soft3_jt1.0+mt0.5_mu0.3_50000/samples_50000x128x128x3.npz'

temp_list = [index for index, value in enumerate(data_path) if value == '/']
if len(temp_list) > 0:
    index = temp_list[-1]
    folder_path = data_path[:index+1]
    file_name = data_path[index+1:]
else:
    folder_path = ''
    file_name = data_path
temp_list = [index for index, value in enumerate(folder_path) if value == '_']
if len(temp_list) > 0:
    index = temp_list[-1]
    folder_path = folder_path[:index+1]

data = np.load(data_path)
for key in data.keys():
    print('key', key)

print('arr_0', data['arr_0'].shape, 'arr_1', data['arr_1'].shape)
samples = data['arr_0']
labels = data['arr_1']

num_sample = min(num_sample, labels.shape[0])
select_row = random.sample(range(samples.shape[0]), num_sample)

samples = samples[select_row]
labels = labels[select_row]
print('samples', samples.shape, 'labels', labels.shape)

out_path = folder_path + str(num_sample) + '_Imagefolder'
os.makedirs(out_path, exist_ok=True)
print('out_path', out_path)

for index, sample in enumerate(samples):
    Image.fromarray(sample).save(f"{out_path}/{index:06d}_{labels[index].item()}.png")
