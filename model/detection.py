#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import shutil
import glob
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import seaborn as sns

rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'WenQuanYi Micro Hei']
rcParams['axes.unicode_minus'] = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"yolo_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def scan_all_labels(base_path, anno_dirs, max_class_idx=15):
    
    all_class_indices = set()
    
    for anno_dir in anno_dirs:
        full_path = os.path.join(base_path, anno_dir)
        if not os.path.exists(full_path):
            logger.warning(f"Directory {full_path} does not exist")
            continue
            
        for txt_file in glob.glob(os.path.join(full_path, "*.txt")):
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:  
                            try:
                                class_idx = int(parts[0])

                                if 0 <= class_idx <= max_class_idx:
                                    all_class_indices.add(class_idx)
                                else:
                                    logger.warning(f"Found out-of-range class index in file {txt_file}: {class_idx}, ignored")
                            except ValueError:
                                logger.warning(f"Found invalid class index in file {txt_file}: {parts[0]}")
            except Exception as e:
                logger.warning(f"Unable to read file {txt_file}: {e}")
    
    return sorted(all_class_indices)

def prepare_dataset(base_path, img_dir, anno_dir, out_dir, class_mapping, max_class_idx=15):
    
    os.makedirs(out_dir, exist_ok=True)
    count_images = 0
    count_labels = 0
    
    img_path = os.path.join(base_path, img_dir)
    if not os.path.exists(img_path):
        logger.error(f"Image directory {img_path} does not exist")
        return 0, 0
    
    images = [f for f in os.listdir(img_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    
    total_images = len(images)
    logger.info(f"Found {total_images} image files in {img_path}")
    
    for i, img_file in enumerate(images):
        if i % 100 == 0:
            logger.info(f"Processing progress: {i}/{total_images} ({i/total_images*100:.1f}%)")
            
        img_name = os.path.splitext(img_file)[0]
        img_full_path = os.path.join(img_path, img_file)
        
        try:
            shutil.copy(img_full_path, os.path.join(out_dir, img_file))
            count_images += 1
        except Exception as e:
            logger.error(f"Unable to copy image {img_full_path}: {e}")
            continue
        
        anno_path = os.path.join(base_path, anno_dir, f"{img_name}.txt")
        if os.path.exists(anno_path):
            try:
                with open(anno_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:  
                        try:
                            old_class = int(parts[0])
                            
                            if 0 <= old_class <= max_class_idx and old_class in class_mapping:
                                parts[0] = str(class_mapping[old_class])
                                new_lines.append(" ".join(parts) + "\n")
                            else:

                                pass
                        except ValueError:
                            logger.warning(f"Found invalid class index in file {anno_path}: {parts[0]}")
                
                if new_lines:
                    with open(os.path.join(out_dir, f"{img_name}.txt"), 'w') as f:
                        f.writelines(new_lines)
                        count_labels += 1
            except Exception as e:
                logger.error(f"Error processing label file {anno_path}: {e}")
                
    return count_images, count_labels

def create_splits(dataset_dir, val_ratio=0.2, seed=42):
    import random
    random.seed(seed)
    
    image_files = [f for f in os.listdir(dataset_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
    
    if not image_files:
        logger.error(f"No image files found in {dataset_dir}")
        return [], []
    
    random.shuffle(image_files)
    
    val_size = max(1, int(len(image_files) * val_ratio))
    
    val_files = image_files[:val_size]
    train_files = image_files[val_size:]
    
    logger.info(f"Created dataset splits: {len(train_files)} train samples, {len(val_files)} validation samples")
    
    return train_files, val_files

def find_annotation_dir(base_path, possible_dirs=None):
    
    if possible_dirs is None:
        possible_dirs = [
            'train/annotation',
        ]
    
    for dir_name in possible_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):

            txt_files = glob.glob(os.path.join(dir_path, "*.txt"))
            if txt_files:
                logger.info(f"Found valid annotation directory: {dir_path}")
                return dir_path
    
    return None

def create_enhanced_yolov11_config(output_dir, num_classes):
    
    config_path = os.path.join(output_dir, 'yolo11_detect.yaml')
    
    yolo11_paths = glob.glob('/root/ultralytics-8.3.27/ultralytics/cfg/models/**/yolo*.yaml', recursive=True)
    base_config_path = None
    
    for path in yolo11_paths:
        if 'yolo11' in path.lower() and 'detect' in path.lower():
            base_config_path = path
            break
    
    if not base_config_path:
        for path in yolo11_paths:
            if 'yolo11' in path.lower() and not any(task in path.lower() for task in ['pose', 'seg', 'cls', 'obb']):
                base_config_path = path
                break
    
    if not base_config_path:
        for path in yolo11_paths:
            if 'yolo11' in path.lower():
                base_config_path = path
                break
                
    if not base_config_path:
        for path in yolo11_paths:
            if 'detect' in path.lower():
                base_config_path = path
                break
    
    if not base_config_path and yolo11_paths:
        base_config_path = yolo11_paths[0]
        
    if not base_config_path:
        logger.error("Cannot find any YOLOv11 configuration file")
        raise FileNotFoundError("Cannot find YOLOv11 configuration file")
    
    logger.info(f"Based on existing detection model configuration: {base_config_path}")
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['nc'] = num_classes
    
    if 'task' in config:
        if config['task'] != 'detect':
            logger.warning(f"Changing task type from {config['task']} to detect")
            config['task'] = 'detect'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    logger.info(f"Successfully created detection model configuration: {config_path}")
    return config_path

def generate_pr_curves_and_metrics(model, data_yaml, save_dir, class_names):

    logger.info("Starting to generate PR curves and calculate class metrics...")
    
    metrics_dir = os.path.join(save_dir, 'evaluation_metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Define IoU thresholds
    iou_thresholds = [0.5, 0.75, 0.9]
    
    all_metrics = {}
    
    for iou_thresh in iou_thresholds:
        logger.info(f"Evaluating IoU threshold = {iou_thresh}")
        
        results = model.val(data=data_yaml, iou=iou_thresh, save_json=True, save_txt=True)
        
        if hasattr(results, 'box'):
            box_metrics = results.box
            
            class_metrics = []
            
            nc = len(class_names)
            
            if hasattr(box_metrics, 'p') and hasattr(box_metrics, 'r'):
                precisions = box_metrics.p  
                recalls = box_metrics.r     
                
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)
                
                ap_per_class = box_metrics.ap if hasattr(box_metrics, 'ap') else None
                
                for i in range(nc):
                    class_name = class_names.get(i, f'class_{i}')
                    
                    metrics_dict = {
                        'Class': class_name,
                        'Class_ID': i,
                        f'Precision@IoU{iou_thresh}': float(precisions[i]) if i < len(precisions) else 0.0,
                        f'Recall@IoU{iou_thresh}': float(recalls[i]) if i < len(recalls) else 0.0,
                        f'F1-Score@IoU{iou_thresh}': float(f1_scores[i]) if i < len(f1_scores) else 0.0,
                    }
                    
                    if ap_per_class is not None and i < len(ap_per_class):
                        metrics_dict[f'AP@IoU{iou_thresh}'] = float(ap_per_class[i])
                    
                    class_metrics.append(metrics_dict)
                
                all_metrics[f'iou_{iou_thresh}'] = class_metrics
                
                generate_pr_curve(box_metrics, iou_thresh, metrics_dir, class_names)
            else:
                logger.warning(f"Unable to get validation metrics for IoU={iou_thresh}")
    
    save_metrics_to_csv(all_metrics, metrics_dir, class_names)
    
    logger.info("PR curves and metrics calculation completed")

def generate_pr_curve(box_metrics, iou_thresh, save_dir, class_names):
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if hasattr(box_metrics, 'curves') and box_metrics.curves is not None:
            pr_curves = box_metrics.curves.get('PR', None)
            
            if pr_curves is not None and len(pr_curves) > 0:
                if len(pr_curves) == 2:
                    recall = pr_curves[0]
                    precision = pr_curves[1]
                    ax.plot(recall, precision, linewidth=2, label=f'mAP@{iou_thresh}')
                
                else:
                    for i, curve in enumerate(pr_curves):
                        if i < len(class_names):
                            class_name = class_names.get(i, f'class_{i}')
                            if len(curve) >= 2:
                                recall = curve[0]
                                precision = curve[1]
                                ax.plot(recall, precision, linewidth=1.5, 
                                       label=f'{class_name}', alpha=0.8)
        
        elif hasattr(box_metrics, 'p_curve') and hasattr(box_metrics, 'r_curve'):

            precision = box_metrics.p_curve
            recall = box_metrics.r_curve
            
            if precision is not None and recall is not None:
                ax.plot(recall, precision, linewidth=2, 
                       label=f'mAP@{iou_thresh}', color='blue')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve @ IoU={iou_thresh}', fontsize=14)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Save figure
        save_path = os.path.join(save_dir, f'PR_curve_iou_{iou_thresh}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved: {save_path}")
        
    except Exception as e:
        logger.error(f"Error generating PR curve (IoU={iou_thresh}): {e}")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if hasattr(box_metrics, 'p') and hasattr(box_metrics, 'r'):

                mean_precision = float(box_metrics.p.mean())
                mean_recall = float(box_metrics.r.mean())
                
                ax.scatter([mean_recall], [mean_precision], s=100, 
                          label=f'Mean (P={mean_precision:.3f}, R={mean_recall:.3f})')
                
                ax.plot([0, mean_recall], [mean_precision, mean_precision], 
                       'k--', alpha=0.3)
                ax.plot([mean_recall, mean_recall], [0, mean_precision], 
                       'k--', alpha=0.3)
            
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(f'Precision-Recall Points @ IoU={iou_thresh}', fontsize=14)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            save_path = os.path.join(save_dir, f'PR_points_iou_{iou_thresh}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Simplified PR plot saved: {save_path}")
            
        except Exception as e2:
            logger.error(f"Generating simplified PR plot also failed: {e2}")

def save_metrics_to_csv(all_metrics, save_dir, class_names):
    try:

        all_data = []
        
        for iou_key, metrics_list in all_metrics.items():
            iou_value = float(iou_key.split('_')[1])
            
            for metric in metrics_list:

                row = {
                    'Class': metric['Class'],
                    'Class_ID': metric['Class_ID'],
                    'IoU_Threshold': iou_value
                }
                
                for key, value in metric.items():
                    if key not in ['Class', 'Class_ID']:

                        if 'Precision' in key:
                            row['Precision'] = value
                        elif 'Recall' in key:
                            row['Recall'] = value
                        elif 'F1-Score' in key:
                            row['F1-Score'] = value
                        elif 'AP' in key:
                            row['AP'] = value
                
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        
        df = df.sort_values(['Class_ID', 'IoU_Threshold'])
        
        detailed_csv_path = os.path.join(save_dir, 'class_metrics_detailed.csv')
        df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Detailed metrics saved: {detailed_csv_path}")
        
        for iou in [0.5, 0.75, 0.9]:
            iou_df = df[df['IoU_Threshold'] == iou].copy()
            if not iou_df.empty:

                iou_df = iou_df.drop('IoU_Threshold', axis=1)
                
                iou_csv_path = os.path.join(save_dir, f'class_metrics_iou_{iou}.csv')
                iou_df.to_csv(iou_csv_path, index=False, encoding='utf-8-sig')
                logger.info(f"Metrics for IoU={iou} saved: {iou_csv_path}")
        
        if 'Precision' in df.columns and 'Recall' in df.columns and 'F1-Score' in df.columns:

            for metric in ['Precision', 'Recall', 'F1-Score', 'AP']:
                if metric in df.columns:
                    pivot_df = df.pivot(index='Class', columns='IoU_Threshold', values=metric)
                    pivot_csv_path = os.path.join(save_dir, f'{metric.lower()}_by_iou.csv')
                    pivot_df.to_csv(pivot_csv_path, encoding='utf-8-sig')
                    logger.info(f"{metric} pivot table saved: {pivot_csv_path}")
        
        summary_data = []
        for iou in [0.5, 0.75, 0.9]:
            iou_df = df[df['IoU_Threshold'] == iou]
            if not iou_df.empty and 'Precision' in iou_df.columns:
                summary_data.append({
                    'IoU_Threshold': iou,
                    'Mean_Precision': iou_df['Precision'].mean(),
                    'Mean_Recall': iou_df['Recall'].mean(),
                    'Mean_F1-Score': iou_df['F1-Score'].mean(),
                    'Mean_AP': iou_df['AP'].mean() if 'AP' in iou_df.columns else np.nan
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = os.path.join(save_dir, 'metrics_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            logger.info(f"Metrics summary saved: {summary_csv_path}")
            
    except Exception as e:
        logger.error(f"Error saving CSV files: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():

    parser = argparse.ArgumentParser(description='YOLO Training Script (Enhanced Version)')
    parser.add_argument('--base_path', type=str, default='/root/autodl-tmp/labelimg', 
                        help='Data directory root path')
    parser.add_argument('--train_img', type=str, default='/root/autodl-tmp/labelimg/train/images',
                        help='Training image directory name')
    parser.add_argument('--train_anno', type=str, default='/root/autodl-tmp/labelimg/train/annotation',
                        help='Training annotation directory name')
    parser.add_argument('--val_img', type=str, default='/root/autodl-tmp/labelimg/val/images',
                        help='Validation image directory name')
    parser.add_argument('--val_anno', type=str, default='/root/autodl-tmp/labelimg/val/annotation',
                        help='Validation annotation directory name')
    parser.add_argument('--output', type=str, default='/root/ultralytics-8.3.27/runs/detect/624',
                        help='Output directory')
    parser.add_argument('--model', type=str, default='/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11.yaml',
                        help='Model configuration file path')
    parser.add_argument('--weights', type=str, default='',
                        help='Pretrained weights path, empty for default weights')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs')
    parser.add_argument('--batch', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=str, default='0',
                        help='Training device, e.g. 0 or 0,1 or cpu')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last training')

    parser.add_argument('--enhancement', type=str, default='fpn_attention', 
                       choices=['none', 'fpn', 'attention', 'fpn_attention'],
                       help='Select model enhancement architecture: none, fpn, attention, fpn_attention')

    parser.add_argument('--download_weights', action='store_true',
                       help='Manually download pretrained weights')

    parser.add_argument('--weights_dir', type=str, default='./weights',
                       help='Pretrained weights storage directory')

    parser.add_argument('--auto_find_annotations', action='store_true', default=True,
                        help='Automatically find annotation directory')
    
    args = parser.parse_args()
    
    logger.info(f"Subdirectories under base path {args.base_path}:")
    subdirs = [d for d in os.listdir(args.base_path) if os.path.isdir(os.path.join(args.base_path, d))]
    if subdirs:
        for d in subdirs:
            logger.info(f" - {d}")
    else:
        logger.warning(f"No subdirectories under base path {args.base_path}")
    
    if args.auto_find_annotations:
        logger.info("Automatically finding training annotation directory...")
        train_anno_dir = find_annotation_dir(args.base_path)
        if train_anno_dir:
            logger.info(f"Found training annotation directory: {train_anno_dir}")
            args.train_anno = train_anno_dir
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)
    dataset_dir = os.path.join(args.output, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    logger.info("Scanning label files to collect class information...")
    all_class_indices = scan_all_labels(args.base_path, [args.train_anno, args.val_anno])
    
    if not all_class_indices:
        logger.error("No valid class indices found in label files")
        return
        
    logger.info(f"Found class indices: {all_class_indices}")
    
    class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(all_class_indices)}
    class_names = {i: f"class_{old_idx}" for old_idx, i in class_mapping.items()}
    
    logger.info(f"Class mapping: {class_mapping}")
    logger.info(f"Total {len(class_mapping)} classes")
    
    logger.info("Preparing training data...")
    train_images, train_labels = prepare_dataset(
        args.base_path, args.train_img, args.train_anno, dataset_dir, class_mapping)
    logger.info(f"Processed {train_images} training images, {train_labels} training labels")
    
    logger.info("Preparing validation data...")
    val_images, val_labels = prepare_dataset(
        args.base_path, args.val_img, args.val_anno, dataset_dir, class_mapping)
    logger.info(f"Processed {val_images} validation images, {val_labels} validation labels")
    
    if train_labels == 0:
        logger.error("No valid training labels found, cannot continue training")
        return
    
    if val_labels == 0:
        logger.warning("No valid validation labels found, will split validation set from training set")
        train_files, val_files = create_splits(dataset_dir)
        
        train_dir = os.path.join(args.output, 'train')
        val_dir = os.path.join(args.output, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        for f in train_files:
            name_base = os.path.splitext(f)[0]
            for ext in ['', '.txt']:
                src = os.path.join(dataset_dir, f"{name_base}{ext}")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(train_dir, f"{name_base}{ext}"))
                    
        for f in val_files:
            name_base = os.path.splitext(f)[0]
            for ext in ['', '.txt']:
                src = os.path.join(dataset_dir, f"{name_base}{ext}")
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(val_dir, f"{name_base}{ext}"))
                    
        dataset_sections = {'train': 'train', 'val': 'val'}
    else:

        dataset_sections = {'train': 'dataset', 'val': 'dataset'}
    
    data_yaml = os.path.join(args.output, 'data.yaml')
    with open(data_yaml, 'w') as f:
        yaml.dump({
            'path': args.output,
            'train': dataset_sections['train'],
            'val': dataset_sections['val'],
            'nc': len(class_names),
            'names': class_names
        }, f)
    
    logger.info(f"Created data configuration file: {data_yaml}")
    
    if args.enhancement != 'none':
        try:
            logger.info(f"Creating enhanced YOLOv11 configuration ({args.enhancement})...")
            args.model = create_enhanced_yolov11_config(args.output, len(class_names), args.enhancement)
        except Exception as e:
            logger.error(f"Error creating enhanced configuration: {e}")
            logger.warning("Will use standard YOLOv11 configuration")

            if not os.path.exists(args.model):
                logger.warning(f"Model configuration file {args.model} does not exist, trying to find other model files")
                possible_paths = glob.glob("/root/ultralytics-8.3.27/ultralytics/cfg/models/**/yolo*.yaml", recursive=True)
                if possible_paths:
                    args.model = possible_paths[0]
                    logger.info(f"Using available model {args.model}")
                else:
                    logger.error("No available model configuration files found")
                    return
    
    pretrained_weights = None
    if args.weights and os.path.exists(args.weights):

        pretrained_weights = args.weights
        logger.info(f"Using specified pretrained weights: {pretrained_weights}")
    else:
        
        model_basename = os.path.basename(args.model)
        model_scale = 'n'  
        
        if 'yolo11' in model_basename.lower():
            for scale in ['n', 's', 'm', 'l', 'x']:
                if f'yolo11{scale}' in model_basename.lower():
                    model_scale = scale
                    break
        
        weight_filename = f"yolo11{model_scale}.pt"
        weight_path = os.path.join(args.weights_dir, weight_filename)
        
        if os.path.exists(weight_path) and os.path.getsize(weight_path) > 1000000:  #
            logger.info(f"Using local pretrained weights: {weight_path}")
            pretrained_weights = weight_path
        elif args.download_weights:

            import urllib.request
            
            weight_url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{weight_filename}"
            logger.info(f"Manually downloading pretrained weights: {weight_url}")
            
            try:

                import requests
                from tqdm import tqdm
                
                response = requests.get(weight_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(weight_path, 'wb') as f, tqdm(
                    desc=weight_filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)
                
                if os.path.exists(weight_path) and os.path.getsize(weight_path) > 1000000:
                    logger.info(f"Pretrained weights downloaded successfully: {weight_path}")
                    pretrained_weights = weight_path
                else:
                    logger.warning("Downloaded file size is abnormal, may be incomplete")
                    logger.warning("Will try to continue with training from scratch")
            except Exception as e:
                logger.error(f"Failed to download pretrained weights: {e}")
                logger.warning("Will try to continue with training from scratch")

                if os.path.exists(weight_path):
                    os.remove(weight_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f"yolo11_{args.enhancement}_{timestamp}"
    save_dir = os.path.join('runs/train', project_name)
    
    try:
        logger.info(f"Loading model configuration {args.model}")
        model = YOLO(args.model)
        
        if pretrained_weights:
            try:
                logger.info(f"Attempting to load pretrained weights: {pretrained_weights}")
                model = model.load(pretrained_weights)
                logger.info("Pretrained weights loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load pretrained weights: {e}")
                logger.info("Will continue with training from scratch")
        
        train_params = {
            'data': data_yaml,                  
            'epochs': args.epochs,              
            'imgsz': args.img_size,             
            'batch': args.batch,                
            'workers': 8,                       
            'device': args.device,              
            
            'plots': True,                      
            'save_period': 10,                  
            'val': True,                        
            'verbose': True,                    
            
            'lr0': 0.002,                       
            'lrf': 0.0001,                      
            'warmup_epochs': 5.0,               
            'cos_lr': True,                     
            
            'optimizer': 'AdamW',               
            'weight_decay': 0.0005,             
            'momentum': 0.937,                  
            
            'patience': 80,                     
            
            'mosaic': 0.7,                      
            'mixup': 0.1,                       
            'copy_paste': 0.0,                  
            'degrees': 10.0,                    
            'translate': 0.1,                   
            'scale': 0.3,                       
            'fliplr': 0.3,                      
            'flipud': 0.0,                      
            'hsv_h': 0.01,                      
            'hsv_s': 0.5,                       
            'hsv_v': 0.3,                       
            
            'perspective': 0.0,                 
            'rect': True,                       
            
            'multi_scale': True,                
            'pretrained': pretrained_weights is not None,  
            'save': True,                       
            'project': 'runs/train',            
            'name': project_name,               
            'exist_ok': False,                  
            'resume': args.resume,              
            'nbs': 64,                          
            
            'amp': False,
        }
        
        logger.info("Starting model training...")
        logger.info(f"Training parameters: {train_params}")
        start_time = time.time()
        
        results = model.train(**train_params)
        
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training completed, time taken: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        logger.info("Validating best model...")
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            best_model = YOLO(best_model_path)
            
            metrics = best_model.val(data=data_yaml)
            logger.info(f"Best model validation metrics: {metrics}")
            
            generate_pr_curves_and_metrics(best_model, data_yaml, save_dir, class_names)
            
            logger.info("Exporting ONNX model...")
            try:
                onnx_path = os.path.join(save_dir, 'weights', 'best.onnx')
                best_model.export(format='onnx', imgsz=args.img_size)
                logger.info(f"ONNX model exported: {onnx_path}")
            except Exception as e:
                logger.error(f"Error exporting ONNX model: {e}")
        else:
            logger.warning(f"Best model file {best_model_path} does not exist")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
if __name__ == "__main__":
    main()