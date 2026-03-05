from .UMFormer import get_cfg as get_UMFormer_cfg
from .My_DINOv3 import get_cfg as get_DINOv3_cfg
from .SegDINO import get_cfg as get_SegDINO_cfg
from .MFNet import get_cfg as get_MFNet_cfg
from .Mask2Former import get_cfg as get_Mask2Former_cfg
from .UNetFormer import get_cfg as get_UNetFormer_cfg
from .FTUNetFormer import get_cfg as get_FTUNetFormer_cfg
from .RS3Mamba import get_cfg as get_RS3Mamba_cfg
from .CMTFNet import get_cfg as get_CMTFNet_cfg
from .MultiSenseSeg import get_cfg as get_MultiSenseSeg_cfg
from .ESANet import get_cfg as get_ESANet_cfg
from .FTransUNet import get_cfg as get_FTransUNet_cfg
from .CMGFNet import get_cfg as get_CMGFNet_cfg
from .SpectralGPT import get_cfg as get_SpectralGPT_cfg
from .SynFSNet import get_cfg as get_SynFSNet_cfg


def get_cfg(model_name=None, dataset_name=None, **kwargs):
    if model_name is None:
        raise ValueError("Model name must be specified")
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    if 'DINOv3' in model_name:
        cfg = get_DINOv3_cfg(model_name, dataset_name, **kwargs)
    elif model_name == 'UMFormer':
        cfg = get_UMFormer_cfg(dataset_name)
    elif model_name == 'SegDINO':
        cfg = get_SegDINO_cfg(dataset_name)
    elif model_name == 'MFNet':
        cfg = get_MFNet_cfg(dataset_name)
    elif model_name == 'Mask2Former':
        cfg = get_Mask2Former_cfg(dataset_name)
    elif model_name == 'UNetFormer':
        cfg = get_UNetFormer_cfg(dataset_name)
    elif model_name == 'FTUNetFormer':
        cfg = get_FTUNetFormer_cfg(dataset_name)
    elif model_name == 'RS3Mamba':
        cfg = get_RS3Mamba_cfg(dataset_name)
    elif model_name == 'CMTFNet':
        cfg = get_CMTFNet_cfg(dataset_name)
    elif model_name == 'MultiSenseSeg':
        cfg = get_MultiSenseSeg_cfg(dataset_name)
    elif model_name == 'ESANet':
        cfg = get_ESANet_cfg(dataset_name)
    elif model_name == 'FTransUNet':
        cfg = get_FTransUNet_cfg(dataset_name)
    elif model_name == 'CMGFNet':
        cfg = get_CMGFNet_cfg(dataset_name)
    elif model_name == 'SynFSNet':
        cfg = get_SynFSNet_cfg(dataset_name)
    else:
        raise ValueError("Model name is not supported")

    return cfg
