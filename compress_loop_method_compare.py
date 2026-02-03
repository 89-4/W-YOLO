import warnings
import os

# 在 Windows cmd / PowerShell 下也可通过环境变量设置，但在脚本里也可以：
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune


# from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
# from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
# from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune


def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path


def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()


if __name__ == '__main__':

    base_param_dict = {
        # origin
        # 'model': r'C:\Users\tiant\Desktop\Papers\CODE\ultralytics-yolo11-20250415\ultralytics-yolo11-main\a_my_results\ablation\yolo11s-EMA-k32\weights\best.pt',
        'model': r'C:\Users\tiant\Desktop\Papers\CODE\ultralytics-yolo11-20250415\ultralytics-yolo11-main\a_my_results\ablation\A\yolo11s-EMA-k3-FDN-NWD-FDN_prune_friendly\weights\best.pt',

        'data': 'my_data_yaml/walnut_30k.yaml',

        'epochs': 100,
        'batch': 8,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'project': 'a_my_results/pruned_new/method_comparison',

        # prune
        # 'prune_method': 'slim',
        'prune_method': 'l1',
        # 'prune_method': 'group_taylor',
        # 'prune_method': 'lamp',

        'global_pruning': False,
        # 'iterative_steps': 1,
        # 'skip_list': 'FocusFeature',
        'reg': 0.0005,
        'sl_epochs': 100,
        'imgsz': 640,
        # 'fraction':0.3,
        'amp': True,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }

    # 📌 不同剪枝方法比较
    prune_method_list = ['slim', 'l1', 'group_taylor', 'lamp']

    # 固定加速比
    fixed_speed_up = 1.6

    for prune_method in prune_method_list:
        print(f"\n🚀 Start pruning & finetune: method={prune_method}, speed_up={fixed_speed_up}")

        param_dict = copy.deepcopy(base_param_dict)

        # 设置剪枝方法和固定加速比
        param_dict['prune_method'] = prune_method
        param_dict['speed_up'] = fixed_speed_up

        # 名称自动匹配，反映剪枝方法
        param_dict['name'] = f'yolov11s-EFN-{prune_method}-su{fixed_speed_up}-gpF'

        # 1️⃣ 剪枝
        prune_model_path = compress(copy.deepcopy(param_dict))

        # 2️⃣ 微调
        finetune(copy.deepcopy(param_dict), prune_model_path)