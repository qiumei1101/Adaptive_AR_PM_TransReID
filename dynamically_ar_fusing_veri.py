import argparse
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.logger import setup_logger
import torchvision.transforms as T
import glob
from config import cfg
from PIL import Image, ImageFile
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from model import make_model

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 10

def main(args):
    # Load and initialize models
    cfg.merge_from_file(args.cfg1)
    model1 = make_model(cfg, num_class=args.num_classes_1, camera_num=args.camera_num_1, view_num=args.view_num)
    model1.load_param(args.weight1)

    cfg.merge_from_file(args.cfg2)
    model2 = make_model(cfg, num_class=args.num_classes_2, camera_num=args.camera_num_2, view_num=args.view_num)
    model2.load_param(args.weight2)

    cfg.merge_from_file(args.cfg3)
    model3 = make_model(cfg, num_class=args.num_classes_3, camera_num=args.camera_num_3, view_num=args.view_num)
    model3.load_param(args.weight3)

    evaluator = R1_mAP_eval(args.num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if args.device:
        model1.to(args.device)
        model2.to(args.device)
        model3.to(args.device)

    model1.eval()
    model2.eval()
    model3.eval()

    SIZE_TEST_1 = args.size_test_1
    SIZE_TEST_2 = args.size_test_2
    SIZE_TEST_3 = args.size_test_3

    val_transforms_1 = T.Compose([
        T.Resize(SIZE_TEST_1),
        T.ToTensor(),
        T.Normalize(mean=args.ori_mean, std=args.ori_mean)
    ])
    val_transforms_2 = T.Compose([
        T.Resize(SIZE_TEST_2),
        T.ToTensor(),
        T.Normalize(mean=args.ori_mean, std=args.ori_mean)
    ])
    val_transforms_3 = T.Compose([
        T.Resize(SIZE_TEST_3),
        T.ToTensor(),
        T.Normalize(mean=args.ori_mean, std=args.ori_mean)
    ])

    img_path_list = []
    for img_ in glob.glob(args.image_query + "/*.jpg"):
        img_path_list.append(img_)
    for img_ in glob.glob(args.image_gallery + "/*.jpg"):
        img_path_list.append(img_)

    tolal_time_cost = []
    for img_ in img_path_list:
        time_start = time.time()
        with torch.no_grad():
            img = Image.open(img_).convert('RGB')
            w, h = img.size
            ar = w / h
            pid = [(int(Path(img_).stem.split("_")[0])),]
            camid = [(int(Path(img_).stem.split("_")[1][1:])),]

            # Process image with different models
            process_img = val_transforms_1(img).unsqueeze(0).to(args.device)
            feat1 = model1(process_img, cam_label=camid, view_label=0)

            process_img = val_transforms_2(img).unsqueeze(0).to(args.device)
            feat2 = model2(process_img, cam_label=camid, view_label=0)

            process_img = val_transforms_3(img).unsqueeze(0).to(args.device)
            feat3 = model3(process_img, cam_label=camid, view_label=0)

            # Calculate weights based on aspect ratio
            weights = []
            model_1_ar = args.model_1_ar
            model_2_ar = args.model_2_ar
            model_3_ar = args.model_3_ar

            weight1 = 1.2 if abs(model_1_ar - ar) <= 0.6 else 1.0 if abs(model_1_ar - ar) <= 1.25 else 0.9
            weight2 = 1.5 if abs(model_2_ar - ar) <= 0.6 else 1.0 if abs(model_2_ar - ar) <= 1.25 else 0.9
            weight3 = 1.8 if abs(model_3_ar - ar) <= 0.6 else 1.0 if abs(model_3_ar - ar) <= 1.25 else 0.9

            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)

            # Combine features
            feat = weights[0] * feat1 + weights[1] * feat2 + weights[2] * feat3

            time_end = time.time()
            tolal_time_cost.append(time_end - time_start)
            evaluator.update((feat, pid, camid))

    print("Time cost average:", sum(tolal_time_cost) / len(img_path_list))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print("Validation Results")
    print("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle ReID Inference Script")
    parser.add_argument("--cfg1", type=str, required=True, help="Path to config file for model 1")
    parser.add_argument("--weight1", type=str, required=True, help="Path to weights file for model 1")
    parser.add_argument("--cfg2", type=str, required=True, help="Path to config file for model 2")
    parser.add_argument("--weight2", type=str, required=True, help="Path to weights file for model 2")
    parser.add_argument("--cfg3", type=str, required=True, help="Path to config file for model 3")
    parser.add_argument("--weight3", type=str, required=True, help="Path to weights file for model 3")
    parser.add_argument("--num_classes_1", type=int, default=576, help="Number of classes for model 1")
    parser.add_argument("--camera_num_1", type=int, default=20, help="Number of cameras for model 1")
    parser.add_argument("--num_classes_2", type=int, default=576, help="Number of classes for model 2")
    parser.add_argument("--camera_num_2", type=int, default=20, help="Number of cameras for model 2")
    parser.add_argument("--num_classes_3", type=int, default=576, help="Number of classes for model 3")
    parser.add_argument("--camera_num_3", type=int, default=20, help="Number of cameras for model 3")
    parser.add_argument("--view_num", type=int, default=1, help="Number of views for models")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device to use for inference")
    parser.add_argument("--num_query", type=int, default=1678, help="Number of queries")
    parser.add_argument("--size_test_1", type=list, default=[224, 224], help="Input size for model 1")
    parser.add_argument("--size_test_2", type=list, default=[224, 212], help="Input size for model 2")
    parser.add_argument("--size_test_3", type=list, default=[224, 298], help="Input size for model 3")
    parser.add_argument("--ori_mean", type=list, default=[0.5, 0.5, 0.5], help="Normalization mean for input images")
    parser.add_argument("--image_query", type=str, required=True, help="Path to query images directory")
    parser.add_argument("--image_gallery", type=str, required=True, help="Path to gallery images directory")
    parser.add_argument("--model_1_ar", type=float, default=1.0, help="Aspect ratio for model 1")
    parser.add_argument("--model_2_ar", type=float, default=212/224, help="Aspect ratio for model 2")
    parser.add_argument("--model_3_ar", type=float, default=298/224, help="Aspect ratio for model 3")

    args = parser.parse_args()
    main(args)
