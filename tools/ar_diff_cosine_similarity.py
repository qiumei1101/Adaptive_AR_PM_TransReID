from scipy import spatial
import torch
import numpy as np
import argparse
from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import os, glob
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 10
torch.cuda.empty_cache()
device = "cuda:3"
from datasets.vehicleid import VehicleID
veh = VehicleID()
train, query, gallery = veh.process_split(True)
# print("train", train)
cfg1 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VehicleID/vit_transreid_stride.yml'
weight1= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/vehicleID_vit_transreid_stride_384x384_patch_mixup_in_pixel/transformer_120.pth'
name_= 'vehicleID_vit_transreid_stride_384x384_patch_mixup_in_pixel'
# cfg1 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VeRi/vit_transreid_stride_test.yml'
# weight1= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/veri_vit_transreid_224x224_patch_mixup_in_pixel/transformer_120.pth'
# name_= 'veri_vit_transreid_stride_224x224_patch_mixup_in_pixel'
num_classes_1 = 13164
camera_num_1 = 1
# num_classes_1 = 576
# camera_num_1 = 20
view_num=1
num_query = len(query)
from scipy.spatial import distance

feats = []
cfg.merge_from_file(cfg1)
model1 = make_model(cfg, num_class=num_classes_1, camera_num=camera_num_1, view_num = view_num)
model1.load_param(weight1)
if device:
    model1.to(device)

model1.eval()
val_transforms_1 = T.Compose([
    T.Resize(cfg.INPUT.SIZE_TRAIN),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

img_path_list = []
camids = []
positive_pairs_sm = []
negative_pairs_sm = []

# image_query = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/image_query'
# image_gallery ='/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/image_test'
# for img_ in glob.glob(image_query+"/*.jpg"):
#     img_path_list.append(img_)
# for img_ in glob.glob(image_gallery+"/*.jpg"):
#     img_path_list.append(img_)
image_query = query

image_gallery = gallery

for i in range(len(image_query)):
    img_path_list.append(image_query[i])
for i in range(len(image_gallery)):
    img_path_list.append(image_gallery[i])
# for img_ in img_path_list:
#     print("img_",img_)
#     with torch.no_grad():
#         img = Image.open(img_[0]).convert('RGB')
#         # frame = cv2.imread(img)
#         w,h = img.size
#         ar = w/h
#         pid =[img_[1]]
#         camid = [img_[2]]
def extract_features(img_path_list, model):
    features = []
    aspect_ratios =[]
    labels = []
    for img_ in img_path_list:
        print("img_",img_)        
        with torch.no_grad():
            img = Image.open(img_[0]).convert('RGB')
            # frame = cv2.imread(img)
            w,h = img.size
            ar = w/h
            pid =img_[1]
            camid = img_[2]
            # pid = [(int(Path(img_).stem.split("_")[0])),]
            # camid = [(int(Path(img_).stem.split("_")[1][1:])),]
            process_img = val_transforms_1(img)
            process_img=process_img[None,]

            process_img = process_img.to(device)
            print("process_img.shape",process_img.shape)
            feat = model(process_img, cam_label=camid, view_label=0)
            features.append(feat.cpu())
            aspect_ratios.append(ar)
            labels.append(pid)
    return torch.cat(features),aspect_ratios,labels

features,aspect_ratios,labels = extract_features(img_path_list, model1)

def calculate_cosine_similarity(features,labels, aspect_ratios):
    similarities_same = []
    similarities_notsame = []
    ar_diff_same = []
    ar_diff_nonsame = []

    for i in range(0, features.size(0)-1, 1):
        sim = distance.cosine(features[i].unsqueeze(0).numpy().flatten(), features[i+1].unsqueeze(0).numpy().flatten())
        # print("features[i].unsqueeze(0)",features[i].unsqueeze(0).numpy().flatten())
        # print("abs(aspect_ratios[i]-aspect_ratios[i+1])",abs(aspect_ratios[i]-aspect_ratios[i+1]))
        if abs(aspect_ratios[i]-aspect_ratios[i+1])>0.6:
            print("abs(aspect_ratios[i]-aspect_ratios[i+1])",abs(aspect_ratios[i]-aspect_ratios[i+1]))
        if labels[i]==labels[i+1]:
             if abs(aspect_ratios[i]-aspect_ratios[i+1])>0.3:
                ar_diff_same.append(abs(aspect_ratios[i]-aspect_ratios[i+1]))
                similarities_same.append(1-sim)
        else:
             if abs(aspect_ratios[i]-aspect_ratios[i+1])<0.3:
                ar_diff_nonsame.append(abs(aspect_ratios[i]-aspect_ratios[i+1]))
                similarities_notsame.append(1-sim)
    return similarities_same,similarities_notsame,ar_diff_same,ar_diff_nonsame

similarities_same,similarities_notsame,ar_diff_same,ar_diff_nonsame = calculate_cosine_similarity(features,labels, aspect_ratios)
# Assume `aspect_ratios` list corresponds to each pair's adjusted aspect ratio
# aspect_ratios = [1.0, 1.5, 2.0] * (len(similarities) // 3)  # Example: cycling through 3 different ratios
plt.scatter(ar_diff_same, similarities_same, color='blue')
plt.xlabel('Aspect Ratio Difference')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity vs. Aspect Ratio for Same Vehicle Pairs',fontweight='bold')
plt.grid(True)
plt.ylim(-1,1)
plt.savefig(str(name_)+"-Cosine Similarity vs. Aspect Ratio for Same Vehicle Pairs.png")
# plt.show()
plt.clf()
plt.scatter(ar_diff_nonsame, similarities_notsame, color='blue')
plt.xlabel('Aspect Ratio Difference')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity vs. Aspect Ratio for Different Vehicle Pairs',fontweight='bold')
plt.grid(True)
plt.ylim(-1,1)

plt.savefig(str(name_)+"-Cosine Similarity vs. Aspect Ratio for Different Vehicle Pairs.png")
# plt.show()
plt.clf()