import torch
import os

# 旧文件，新版本
models_old_path = 'models_old'
# 输出文件，旧版本
models_new_path = 'models_new'

for i in os.listdir(models_old_path):
    state_dict = torch.load(os.path.join(models_old_path, i), map_location="cpu")
    torch.save(state_dict, os.path.join(models_new_path, i), _use_new_zipfile_serialization=False)
    print(i, 'saved')