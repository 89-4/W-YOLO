import warnings

warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
import pandas as pd
import time


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def save_to_excel(data, excel_path='validation_walnut_30k.xlsx'):
    """将结果保存到Excel文件，如果文件存在则追加"""
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        new_df = pd.DataFrame([data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([data])

    df.to_excel(excel_path, index=False)
    print(f'结果已保存至 {excel_path}')


def validate_model(model_path, model_name):
    """验证单个模型并返回结果"""
    print(f"\n{'=' * 50}")
    print(f'正在验证模型: {model_name}')
    print(f'模型路径: {model_path}')
    print(f"{'=' * 50}")

    start_time = time.time()
    model = YOLO(model_path)
    result = model.val(
        data='../my_data_yaml/walnut_30k.yaml',
        split='val',
        imgsz=640,
        batch=16,
        iou=0.6,
        save_json=True,  # 保存JSON格式的检测结果
        save_txt=True,  # 保存TXT格式的标注
        save_conf=True,  # 在TXT中保存置信度
          # 保存JSON结果
        project='av_my_vals/walnut_30k/AFAF',
        name=f'exp_{model_name}',
        verbose=True,

    )

    if model.task == 'detect':
        n_l, n_p, n_g, flops = model_info(model.model)

        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        fps_overall = 1000 / all_time_per_image
        fps_inference = 1000 / inference_time_per_image

        model_names = list(result.names.values())
        try:
            covered_idx = model_names.index('covered_walnut')
            walnut_idx = model_names.index('walnut')
        except ValueError:
            print("警告：未找到'covered_walnut'或'walnut'类别，将使用第一个类别")
            covered_idx = walnut_idx = 0

        # 你的类别名列表
        class_names = ['Front-lit Clear', 'Backlit Clear', 'Front-lit Occluded', 'Backlit Occluded']

        # 基础性能指标
        excel_data = {
            'Model_Name': model_name,
            'GFLOPs': f'{flops:.1f}',
            'Parameters': n_p,
            'FPS_overall': f'{fps_overall:.2f}',
            'FPS_inference': f'{fps_inference:.2f}',
            'all_P': f"{result.results_dict['metrics/precision(B)']:.4f}",
            'all_R': f"{result.results_dict['metrics/recall(B)']:.4f}",
            'all_mAP': f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
            'all_mAP50': f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            'all_mAP50_90': f"{np.mean(result.box.all_ap[:, :9]):.4f}",
            'Model_Size_MB': get_weight_size(model_path),
            'Validation_Time': f"{time.time() - start_time:.1f}s"
        }

        # 🔹为每个类别分别添加指标
        for idx, cname in enumerate(class_names):
            excel_data[f'{cname}_P'] = f"{result.box.p[idx]:.4f}"
            excel_data[f'{cname}_R'] = f"{result.box.r[idx]:.4f}"
            excel_data[f'{cname}_mAP'] = f"{result.box.ap[idx]:.4f}"
            excel_data[f'{cname}_mAP50'] = f"{result.box.ap50[idx]:.4f}"
            excel_data[f'{cname}_mAP50_90'] = f"{np.mean(result.box.all_ap[idx, :9]):.4f}"

        # excel_data = {
        #     'Model_Name': model_name,
        #     'GFLOPs': f'{flops:.1f}',
        #     'Parameters': n_p,
        #     'FPS_overall': f'{fps_overall:.2f}',
        #     'FPS_inference': f'{fps_inference:.2f}',
        #
        #     # 'covered_walnut_P': f"{result.box.p[covered_idx]:.4f}",
        #     # 'covered_walnut_R': f"{result.box.r[covered_idx]:.4f}",
        #     # 'covered_walnut_mAP': f"{result.box.ap[covered_idx]:.4f}",
        #     # 'covered_walnut_mAP50': f"{result.box.ap50[covered_idx]:.4f}",
        #     # 'covered_walnut_mAP50_90': f"{np.mean(result.box.all_ap[covered_idx, :9]):.4f}",
        #     #
        #     # 'walnut_P': f"{result.box.p[walnut_idx]:.4f}",
        #     # 'walnut_R': f"{result.box.r[walnut_idx]:.4f}",
        #     # 'walnut_mAP': f"{result.box.ap[walnut_idx]:.4f}",
        #     # 'walnut_mAP50': f"{result.box.ap50[walnut_idx]:.4f}",
        #     # 'walnut_mAP50_90': f"{np.mean(result.box.all_ap[walnut_idx, :9]):.4f}",
        #
        #     'all_P': f"{result.results_dict['metrics/precision(B)']:.4f}",
        #     'all_R': f"{result.results_dict['metrics/recall(B)']:.4f}",
        #     'all_mAP': f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
        #     'all_mAP50': f"{result.results_dict['metrics/mAP50(B)']:.4f}",
        #     'all_mAP50_90': f"{np.mean([np.mean(result.box.all_ap[i, :9]) for i in range(len(model_names))]):.4f}",
        #     'Model_Size_MB': get_weight_size(model_path),
        #     'Validation_Time': f"{time.time() - start_time:.1f}s"
        # }

        summary_table = PrettyTable()
        summary_table.title = f"模型摘要 - {model_name}"
        summary_table.field_names = ["指标", "值"]
        for key, value in excel_data.items():
            if key not in ['Model_Name', 'Validation_Time']:
                summary_table.add_row([key, value])
        print(summary_table)

        return excel_data
    return None


if __name__ == '__main__':
    # 配置路径，可以是文件夹或单个 .pt 文件
    #base_dir = '../a_my_results/yolo11s-C3k2_EMA-FPNs/yolo11s-C3k2_EMA-ASF/weights/best.pt'
    #base_dir = '../a_my_results/yolo11s-C3k2_EMA-FPNs/yolo11s-C3k2_EMA-bifpn-GLSA/weights/best.pt'
    #base_dir = '../a_my_results/yolo11s-C3k2_EMA-FPNs/yolo11s-C3k2_EMA-CGAFusion/weights/best.pt'

    #base_dir = '../a_my_results/ablation/A/yolo11s-EMA-k3-FDN-NWD-FDN_prune_friendly/weights/best.pt'

    # method compare
    #base_dir = '../a_my_results/pruned_new/method_comparison/yolov11s-EFN-slim-su1.6-gpF-finetune/weights/best.pt'
    base_dir = '../a_my_results/pruned_new/method_comparison/gh/yolov11s-EFN-group_hessian-su1.6-gpF-finetune/weights/best.pt'
    #base_dir = '../a_my_results/pruned_new/method_comparison/yolov11s-EFN-group_taylor-su1.6-gpF-finetune2/weights/best.pt'
    #base_dir = '../a_my_results/pruned_new/method_comparison/yolov11s-EFN-lamp-su1.6-gpF-finetune/weights/best.pt'

    #taylor group compare
    #base_dir = '../a_my_results/pruned_new/l1/l1_100/yolov11s-EFN-l1-su1.2-gpF-finetune2/weights/best.pt'
    #base_dir = '../a_my_results/pruned_new/l1/l1_100/yolov11s-EFN-l1-su1.4-gpF-finetune/weights/best.pt'
    #base_dir = '../a_my_results/pruned_new/method_comparison/yolov11s-EFN-group_taylor-su1.6-gpF-finetune2/weights/best.pt'
    #base_dir = '../a_my_results/pruned_new/l1/l1_100/yolov11s-EFN-l1-su1.8-gpF-finetune/weights/best.pt'
    #base_dir = '../a_my_results/pruned_new/l1/l1_100/yolov11s-EFN-l1-su2.0-gpF-finetune/weights/best.pt'





    #base_dir = '../a_my_results/a_custom_yolo11_walnut_30k_results/yolo11s/weights/best.pt'
    #base_dir = '../a_my_results/a_custom_yolo11_walnut_30k_results/yolo11s_30k_bifpn2/weights/best.pt'

    #base_dir = '../a_my_results/yolo11s-attention-IoU'
    #base_dir = 'a_papers_yolo11_train_walnut_onetag_batchsize_8_optimizer_SGD/yolo11-C3k2-I/weights/best.pt'
    excel_path = '../a_val_results_xlsx/val_prune_method_compare.xlsx'
    #excel_path = '../a_val_results_xlsx/val_prune_group_taylor.xlsx'

    # 判断 base_dir 是文件还是目录
    if os.path.isfile(base_dir) and base_dir.endswith('.pt'):
        # 单个模型验证并追加结果
        model_path = base_dir
        # 从路径中提取模型目录名，而非文件名
        parent = os.path.basename(os.path.dirname(model_path))
        if parent.lower() == 'weights':
            # 向上一级获取实际模型名称
            model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        else:
            # 若目录并非 "weights"，直接使用父目录名
            model_name = parent
        try:
            result_data = validate_model(model_path, model_name)
            if result_data:
                save_to_excel(result_data, excel_path)
        except Exception as e:
            print(f"验证模型 {model_name} 时出错: {str(e)}")
    else:
        # 目录下批量验证并追加结果
        for model_dir in os.listdir(base_dir):
            model_path = os.path.join(base_dir, model_dir, 'weights', 'best.pt')

            if os.path.exists(model_path):
                try:
                    result_data = validate_model(model_path, model_dir)
                    if result_data:
                        save_to_excel(result_data, excel_path)
                except Exception as e:
                    print(f"验证模型 {model_dir} 时出错: {str(e)}")
            else:
                print(f"跳过 {model_dir} - 未找到模型文件")

    print(f"\n{'=' * 50}")
    print(f"所有模型验证完成! 结果已保存至: {excel_path}")
    print(f"{'=' * 50}")
