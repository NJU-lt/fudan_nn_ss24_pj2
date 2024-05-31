from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import random
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# 可视化
def visualize_bboxes(image,image_name, bboxes, labels, scores, label_show = True):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, label, score in zip(bboxes, labels, scores):
        x_min, y_min, x_max, y_max = bbox.tolist()
        score_value = f"{class_names[label.item()]}:{score.item():.2f}"
        # Draw bounding box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        if label_show:
            ax.text(x_min, y_min - 12, score_value, bbox=dict(facecolor='black', alpha=0.5), fontsize=8, color='white')

    plt.axis('off')
    plt.savefig(image_name)

# 初始化 Faster R-CNN 模型
faster_rcnn_config_file = '/root/Test/MM_detection/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_voc/faster_rcnn_r50_fpn_1x_voc.py'
faster_rcnn_checkpoint_file = '/root/Test/MM_detection/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_voc/epoch_16.pth'
faster_rcnn_model = init_detector(faster_rcnn_config_file, faster_rcnn_checkpoint_file, device='cuda:0')
# 图片路径
img_dir = "data/coco/test2017/"
img_list = os.listdir(img_dir)
# 可视化
pred_score_thr = 0.3 # score阈值
class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
for idx,img in enumerate(img_list):
    if idx>4:
        break
    random_idx = random.randint(0,len(img_list))
    img = img_dir+img_list[random_idx]
    # 推断模型
    faster_rcnn_result = inference_detector(faster_rcnn_model, img)
    # 读取图片
    image = Image.open(img)
    image_np = np.array(image)
    # 可视化
    # 第一阶段产生的proposal box
    pred_instances = faster_rcnn_result.rpn_res
    pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
    visualize_bboxes(image_np, 'outputs/faster_rcnn_first.png', pred_instances.bboxes, pred_instances.labels, pred_instances.scores, False)
    # 最终预测结果
    pred_instances = faster_rcnn_result.pred_instances
    pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
    visualize_bboxes(image_np, 'outputs/faster_rcnn_final.png', pred_instances.bboxes, pred_instances.labels, pred_instances.scores)
    # 图片对比
    faster_rcnn_image = Image.open('outputs/faster_rcnn_first.png')
    yolo_image = Image.open('outputs/faster_rcnn_final.png')
    border_width = 20
    combined_width = faster_rcnn_image.width + yolo_image.width + border_width*3
    combined_height = max(faster_rcnn_image.height, yolo_image.height) + border_width*2
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

    draw = ImageDraw.Draw(combined_image)
    # 将 Faster R-CNN 的结果绘制在左侧
    combined_image.paste(faster_rcnn_image, (border_width, border_width))
    # 将 YOLO 的结果绘制在右侧
    combined_image.paste(yolo_image, (faster_rcnn_image.width + 2*border_width, border_width))

    draw.text((border_width, 0), "First Stage", fill="black")
    draw.text((faster_rcnn_image.width + 2*border_width,0), "Final Prediction", fill="black")

    # 保存合并后的图片
    combined_image.save('outputs/first_second_'+str(idx)+'.png')
