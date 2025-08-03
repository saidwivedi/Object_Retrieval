import torch
from collections import OrderedDict
import re
from huggingface_hub import hf_hub_download
import torch.nn as nn

from .ppat import PointPatchTransformer, Projected



def module(state_dict: dict, name):
    return {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if k.startswith(name + '.')}

def G14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 512*3, 256, 384, 0.2, 64, 6),
        nn.Linear(512, 1280)
    )
    model.load_state_dict(module(s['state_dict'], 'module'))
    return model

def L14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6),
        nn.Linear(512, 768)
    )
    model.load_state_dict(module(s, 'pc_encoder'))
    return model

def B32(s):
    model = PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6)
    model.load_state_dict(module(s, 'pc_encoder'))
    return model

model_list = {
    "OpenShape/openshape-pointbert-vitb32-rgb": B32,
    "OpenShape/openshape-pointbert-vitl14-rgb": L14,
    "OpenShape/openshape-pointbert-vitg14-rgb": G14,
}

def load_spconv_model(model_name="OpenShape/openshape-spconv-shapenet-only"):
    import MinkowskiEngine as ME
    from .Minkowski import MinkResNet34

    model = MinkResNet34()
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only

    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    model.load_state_dict(model_dict)
    model.eval()
    return model

def load_pointbert_model(model_name="OpenShape/openshape-pointbert-vitg14-rgb"):
    s = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model = model_list[model_name](s).eval()
    return model

def load_model(model_name):
    if 'spconv' in model_name:
        return load_spconv_model(model_name)
    elif 'pointbert' in model_name:
        return load_pointbert_model(model_name)

def get_model_name(model_name, model_type):
    if model_name is None:
        if model_type == 'shapenet':
            model_name = f'OpenShape/openshape-spconv-shapenet-only'
            model_arch = 'spconv'
        elif model_type == 'objaverse':
            model_name = f'OpenShape/openshape-pointbert-vitg14-rgb'
            model_arch = 'pointbert'
        else:
            raise ValueError(f'Invalid model type: {model_type}')
        print(f'\nUsing model {model_name} which is trained on {model_type.capitalize()} dataset')
    else:
        if 'spconv' in model_name:
            model_arch = 'spconv'
        elif 'pointbert' in model_name:
            model_arch = 'pointbert'
        else:
            raise ValueError(f'Invalid model name: {model_name}')
        print(f'\nUsing model {model_name}')


    return model_name, model_arch