# 第二部分环境配置和指令

## 安装mmdetection
conda create -n mmdet python=3.10 -y
conda activate mmdet
pip install torch torchvision torchaudio
pip install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .

## faster rcnn 训练
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc.py

## faster rcnn 测试
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc.py
work_dirs/faster_rcnn_r50_fpn_1x_voc/epoch_16.pth --out results.pkl

## yolov3 训练
python tools/train.py configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py

## yolov3 测试
python tools/test.py configs/yolo/yolov3_d53_8xb8-ms-416-273e_coco.py
work_dirs/yolov3_d53_8xb8-ms-416-273e_coco/epoch_196.pth --out
results_yolov3.pkl

## 选择测试集中几张图片并绘制proposals和检测结果
python task2.py

## 可视化并比较faster rcnn和yolov3结果
python task3.py
