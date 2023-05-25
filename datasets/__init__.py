import torch.utils.data
import torchvision

from .ytvos import build as build_ytvos
from .davis import build as build_davis
from .refexp import build as build_refexp
from .refexp2seq import build as build_seq_refexp
from .concat_dataset import build as build_joint


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'ytvos':
        return build_ytvos(image_set, args)
    if dataset_file == 'davis':
        return build_davis(image_set, args)
    if dataset_file == "refcoco" or dataset_file == "refcoco+" or dataset_file == "refcocog":
        return build_seq_refexp(dataset_file, image_set, args)
    # for joint training of refcoco and ytvos
    if dataset_file == 'joint':
        return build_joint(image_set, args)
    raise ValueError(f'dataset {dataset_file} not supported')
