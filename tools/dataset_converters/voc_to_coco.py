import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Pascal VOC to COCO format')
    parser.add_argument('devkit_path', help='Pascal VOC dataset path')
    parser.add_argument('--out-dir', help='Output directory for COCO dataset')
    parser.add_argument('--out-format', default='coco', help='Output format (default: coco)')
    args = parser.parse_args()
    return args

def get_image_info(annotation_root, image_id):
    filename = annotation_root.findtext('filename')
    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))
    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': image_id
    }
    return image_info

def get_annotation_info(obj, label2id, image_id, ann_id):
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin'))
    ymin = int(bndbox.findtext('ymin'))
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    category = obj.findtext('name')
    category_id = label2id[category]
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'id': ann_id,
        'image_id': image_id,
        'iscrowd': 0
    }
    return ann

def convert_voc_to_coco(devkit_path, year, split, output_file):
    label2id = defaultdict(int)
    annotation_id = 1
    image_id = 1
    annotations = []
    images = []
    categories = []
    # if year == ['2007','2012']:
    #     ann_dir = os.path.join(devkit_path, 'VOC' + "0712", 'Annotations')
    #     img_set_file = os.path.join(devkit_path, 'VOC' + "0712", 'ImageSets', 'Main', split + '.txt')
    # else:
    ann_dir = os.path.join(devkit_path, 'VOC' + year, 'Annotations')
    img_set_file = os.path.join(devkit_path, 'VOC' + year, 'ImageSets', 'Main', split + '.txt')
    with open(img_set_file) as f:
        image_ids = f.read().strip().split()

    for image_id in tqdm(image_ids):
        ann_path = os.path.join(ann_dir, image_id + '.xml')
        ann_tree = ET.parse(ann_path)
        ann_root = ann_tree.getroot()

        image_info = get_image_info(ann_root, image_id)
        images.append(image_info)

        for obj in ann_root.findall('object'):
            category = obj.findtext('name')
            if category not in label2id:
                label2id[category] = len(label2id) + 1
                categories.append({
                    'id': label2id[category],
                    'name': category,
                    'supercategory': 'none'
                })
            annotation_info = get_annotation_info(obj, label2id, image_id, annotation_id)
            annotations.append(annotation_info)
            annotation_id += 1

    coco_format_json = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(output_file, 'w') as f:
        json.dump(coco_format_json, f, indent=4)

def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    years = []
    if os.path.isdir(os.path.join(devkit_path, 'VOC2007')):
        years.append('2007')
    if os.path.isdir(os.path.join(devkit_path, 'VOC2012')):
        years.append('2012')
    if '2007' in years and '2012' in years:
        years.append(['2007', '2012'])
    if not years:
        raise IOError(f'The devkit path {devkit_path} contains neither '
                      '"VOC2007" nor "VOC2012" subfolder')
    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'
    for year in years:
        if year == '2007':
            prefix = 'voc07'
        elif year == '2012':
            prefix = 'voc12'
        elif year == ['2007', '2012']:
            prefix = 'voc0712'
        for split in ['train', 'val', 'trainval']:
            dataset_name = prefix + '_' + split
            print(f'processing {dataset_name} ...')
            convert_voc_to_coco(devkit_path, year, split,
                                os.path.join(out_dir, dataset_name + out_fmt))
        # Skip 'test' for VOC2012 as it doesn't have test.txt
        if year == '2007':
            dataset_name = prefix + '_test'
            print(f'processing {dataset_name} ...')
            convert_voc_to_coco(devkit_path, year, 'test',
                                os.path.join(out_dir, dataset_name + out_fmt))
    print('Done!')

if __name__ == '__main__':
    main()