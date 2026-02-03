import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()
# 预测框粗细和颜色修改问题可看<新手推荐学习视频.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('../a_my_results/pruned_new/method_comparison/yolov11s-EFN-group_taylor-su1.6-gpF-finetune2/weights/best.pt') # select your model.pt path
    #model = YOLO('../a_my_results/a_custom_yolo11_walnut_30k_results/yolo11s/weights/best.pt')

    # 循环处理1-4共4个类别
    for class_id in [0,1, 2, 3]:
        model.predict(source='f:/walnut/walnut_30k/test/P1_00025.jpg',
                      imgsz=1024,
                      project='runs/paper/vis/25',
                      name=f'W-yolo11s_p25_c{class_id}',  # 使用f-string动态生成名称
                      save=True,
                      classes=[class_id],  # 循环变量作为类别参数
                      line_width=2,
                      visualize=True
                      )

