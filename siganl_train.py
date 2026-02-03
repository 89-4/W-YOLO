
import warnings
import os


from sched import scheduler

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from networkx.classes import freeze
import warnings

warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
#from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
#from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
#from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune


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

warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':

	model = YOLO('../a_my_custom_yamls/yolo11s-ablation/yolo11s-C3k2-DAA-FDN.yaml')  # 修改yaml
	# model = YOLO('../a_my_custom_yamls/yolo11s-ablation/yolo11s-C3k2-DAA-FDN.yaml')  # 修改yaml

	# model = YOLO('a_my_custom_yamls/yolo11_org_p2.yaml/yolo11_org_nop5.yaml')
	model.load('../yolo11s.pt')  # 加载预训练权重
	model.train(data='my_data_yaml/walnut_30k.yaml',  # 数据集yaml文件
				imgsz=640,
				epochs=200,
				batch=8,  # 这里的 batch=16 只影响训练（train）阶段的 DataLoader。
				# 验证（val）阶段如果不单独指定 batch，则会使用 Ultralytics 的默认值——32
				workers=8,
				device=0,  # 没显卡则将0修改为'cpu'
				optimizer='SGD',
				amp=True,
				cache=False,  # 服务器可设置为True，训练速度变快
				name='yolo11s-EMA-k3-FDN-NWD-FDN_prune_friendly',
				# name='yolo11s-DAA-FDN',
				# name='yolo11s-DAA-FDN-MWD',
				project='../a_my_results/ablation/A',
				patience=15
				)




	base_param_dict = {
		# origin
		'model': r'C:\Users\tiant\Desktop\Papers\CODE\ultralytics-yolo11-20250415\ultralytics-yolo11-main\a_my_results\ablation\A\yolo11s-EMA-k3-FDN-NWD-FDN_prune_friendly\weights\best.pt',
		'data': 'my_data_yaml/walnut_30k.yaml',

		'epochs': 60,
		'batch': 8,
		'workers': 8,
		'cache': False,
		'optimizer': 'SGD',
		'device': '0',
		'close_mosaic': 0,
		'project': 'a_my_results/pruned_new',

		# prune
		#'prune_method': 'slim',
		#'prune_method': 'l1',
		#'prune_method': 'group_hessian',
		'prune_method': 'lamp',

		'global_pruning': False,
		#'iterative_steps': 1,
		#'skip_list': 'FocusFeature',
		'reg': 0.0005,
		'sl_epochs': 100,
		'imgsz': 640,
		#'fraction':0.3,
		'amp':True,
		'sl_hyp': '../ultralytics/cfg/hyp.scratch.sl.yaml',
		'sl_model': None,
	}

	# 📌 不同加速比
	speed_up_list = [1.1,1.2,1.4,1.6,1.8,2.0]

	for su in speed_up_list:
		print(f"\n🚀 Start pruning & finetune: speed_up={su}")

		param_dict = copy.deepcopy(base_param_dict)

		# 设置 speed_up
		param_dict['speed_up'] = su

		# 名称自动匹配
		param_dict['name'] = f'yolov11s-EFN-lamp-su{su}-gpF'

		# 1️⃣ 剪枝
		prune_model_path = compress(copy.deepcopy(param_dict))

		# 2️⃣ 微调
		finetune(copy.deepcopy(param_dict), prune_model_path)



	# model = YOLO('../a_my_custom_yamls/yolo11s-ablation/yolo11s-C3k2-EMA.yaml')  # 修改yaml
	# # model = YOLO('../a_my_custom_yamls/yolo11s-ablation/yolo11s-C3k2-DAA-FDN.yaml')  # 修改yaml
	#
	# # model = YOLO('a_my_custom_yamls/yolo11_org_p2.yaml/yolo11_org_nop5.yaml')
	# model.load('../yolo11s.pt')  # 加载预训练权重
	# model.train(data='my_data_yaml/walnut_30k.yaml',  # 数据集yaml文件
	# 			imgsz=640,
	# 			epochs=200,
	# 			batch=8,  # 这里的 batch=16 只影响训练（train）阶段的 DataLoader。
	# 			# 验证（val）阶段如果不单独指定 batch，则会使用 Ultralytics 的默认值——32
	# 			workers=8,
	# 			device=0,  # 没显卡则将0修改为'cpu'
	# 			optimizer='SGD',
	# 			amp=True,
	# 			cache=False,  # 服务器可设置为True，训练速度变快
	# 			name='yolo11s-EMA-k3',
	# 			# name='yolo11s-DAA-FDN',
	# 			# name='yolo11s-DAA-FDN-MWD',
	# 			project='../a_my_results/ablation',
	# 			patience=15
	# 			)
	#
	#
	# model = YOLO('../a_my_custom_yamls/yolo11s-ablation/yolo11-C3k2-CBAM.yaml')  # 修改yaml
	# # model = YOLO('a_my_custom_yamls/yolo11_org_p2.yaml/yolo11_org_nop5.yaml')
	# model.load('../yolo11s.pt')  # 加载预训练权重
	# model.train(data='my_data_yaml/walnut_30k.yaml',  # 数据集yaml文件
	# 			imgsz=640,
	# 			epochs=200,
	# 			batch=8,  # 这里的 batch=16 只影响训练（train）阶段的 DataLoader。
	# 			# 验证（val）阶段如果不单独指定 batch，则会使用 Ultralytics 的默认值——32
	# 			workers=8,
	# 			device=0,  # 没显卡则将0修改为'cpu'
	# 			optimizer='SGD',
	# 			amp=True,
	# 			cache=False,  # 服务器可设置为True，训练速度变快
	# 			name='yolo11s-CBAM_og',
	# 			# name='yolo11s-DAA-FDN',
	# 			# name='yolo11s-DAA-FDN-MWD',
	# 			project='../a_my_results/ablation',
	# 			patience=15
	# 			)
	#
	#
	# #
	#
	#
	#
	# #odel = YOLO('../a_my_custom_yamls/yamls_walnut_30k/yolo11-C3k2-EMA.yaml')  # 修改yaml
