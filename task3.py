from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import random
import os
from PIL import Image, ImageDraw

# 初始化 Faster R-CNN 模型
faster_rcnn_config_file = '/root/Test/MM_detection/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_voc/faster_rcnn_r50_fpn_1x_voc.py'
faster_rcnn_checkpoint_file = '/root/Test/MM_detection/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_voc/epoch_16.pth'
faster_rcnn_model = init_detector(faster_rcnn_config_file, faster_rcnn_checkpoint_file, device='cuda:0')
# 初始化 YOLO 模型
yolo_config_file = '/root/Test/MM_detection/mmdetection/work_dirs/yolov3_d53_8xb8-ms-416-273e_coco/yolov3_d53_8xb8-ms-416-273e_coco.py'
yolo_checkpoint_file = '/root/Test/MM_detection/mmdetection/work_dirs/yolov3_d53_8xb8-ms-416-273e_coco/epoch_196.pth'
yolo_model = init_detector(yolo_config_file, yolo_checkpoint_file, device='cuda:0')
# 图片路径
img_dir = "test/"
img_list = os.listdir(img_dir)
# 图片压缩
def compress_image(image_path, max_size=1000):
    # 打开图片
    img = Image.open(image_path)
    width, height = img.size
    if width > max_size or height > max_size:
        if width >= height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img.save(image_path)  # 覆盖原始图片


for idx,img in enumerate(img_list):
    img = img_dir+img
    compress_image(img, max_size=1000)
    # 推断模型
    faster_rcnn_result = inference_detector(faster_rcnn_model, img)
    yolo_result = inference_detector(yolo_model, img)
    # 读取图片
    image = mmcv.imread(img)
    # visualizer可视化
    visualizer = VISUALIZERS.build(faster_rcnn_model.cfg.visualizer)
    visualizer.dataset_meta = faster_rcnn_model.dataset_meta
    visualizer.add_datasample(
        'result',
        image,
        data_sample=faster_rcnn_result,
        draw_gt=False,
        wait_time=0,
        out_file='outputs/faster_rcnn_result.png'
    )
    visualizer = VISUALIZERS.build(yolo_model.cfg.visualizer)
    visualizer.dataset_meta = yolo_model.dataset_meta
    visualizer.add_datasample(
        'result',
        image,
        data_sample=yolo_result,
        draw_gt=False,
        wait_time=0,
        out_file='outputs/yolo_result.png'
    )

    # 图片对比
    faster_rcnn_image = Image.open('outputs/faster_rcnn_result.png')
    yolo_image = Image.open('outputs/yolo_result.png')
    border_width = 20
    combined_width = faster_rcnn_image.width + yolo_image.width + border_width*3
    combined_height = max(faster_rcnn_image.height, yolo_image.height) + border_width*2
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

    draw = ImageDraw.Draw(combined_image)
    # 将 Faster R-CNN 的结果绘制在左侧
    combined_image.paste(faster_rcnn_image, (border_width, border_width))
    # 将 YOLO 的结果绘制在右侧
    combined_image.paste(yolo_image, (faster_rcnn_image.width + 2*border_width, border_width))

    draw.text((border_width, 0), "Faster R-CNN", fill="black")
    draw.text((faster_rcnn_image.width + 2*border_width,0), "YOLO", fill="black")

    # 保存合并后的图片
    combined_image.save('outputs/combined_result'+str(idx)+'.png')