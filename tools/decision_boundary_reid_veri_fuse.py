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
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 10
torch.cuda.empty_cache()
device = "cuda:3"
cfg1 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VeRi/vit_transreid_stride.yml'
weight1= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/veri_vit_transreid_224x224_patch_mixup_in_pixel/transformer_120.pth'

cfg2 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VeRi/vit_transreid_stride_2.yml'
weight2= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/veri_vit_transreid_stride_224x212_patch_mixup_in_pixel/transformer_120.pth'

cfg3 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VeRi/vit_transreid_stride_1.yml'
weight3= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/veri_vit_transreid_stride_224x298_patch_mixup_in_pixel_seed_3407/transformer_120.pth'
name_ = 'VeRi_fused'
# if cam_type == 'VeRi':
num_classes_1 = 576
camera_num_1 = 20
num_classes_2 = 576
camera_num_2 = 20
num_classes_3 = 576
camera_num_3 = 20
view_num = 1
num_query = 1678
#square
cfg.merge_from_file(cfg1)
model1 = make_model(cfg, num_class=num_classes_1, camera_num=camera_num_1, view_num = view_num)
model1.load_param(weight1)

#H>w, ar<1
cfg.merge_from_file(cfg2)
model2 = make_model(cfg, num_class=num_classes_2, camera_num=camera_num_2, view_num = view_num)
model2.load_param(weight2)


#H<w, ar>1
cfg.merge_from_file(cfg3)

model3 = make_model(cfg, num_class=num_classes_3, camera_num=camera_num_3, view_num = view_num)
model3.load_param(weight3)

if device:
  
    model1.to(device)
    model2.to(device)
    model3.to(device)
   
#     # model3= nn.DataParallel(model3,device_ids = [2, 3])
#     model3.to(device3)
model1.eval()
model2.eval()
model3.eval()
SIZE_TEST_1 =  [224, 224]
SIZE_TEST_2 =  [224, 212]
SIZE_TEST_3 =  [224, 298]
mean_=[0.485, 0.456, 0.406]
std_=[0.229, 0.224, 0.225]
ori_mean = [0.5, 0.5, 0.5]
val_transforms_1 = T.Compose([
    T.Resize(SIZE_TEST_1),
    T.ToTensor(),
    T.Normalize(mean=ori_mean, std=ori_mean)
])
val_transforms_2 = T.Compose([
    T.Resize(SIZE_TEST_2),
    T.ToTensor(),
    T.Normalize(mean=ori_mean, std=ori_mean)
])
val_transforms_3 = T.Compose([
    T.Resize(SIZE_TEST_3),
    T.ToTensor(),
    T.Normalize(mean=ori_mean, std=ori_mean)
])


img_path_list = []
camids = []
positive_pairs_sm = []
negative_pairs_sm = []
image_query = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/small_veri'
image_gallery ='/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/image_test'
for img_ in glob.glob(image_query+"/*.jpg"):
    img_path_list.append(img_)
# for img_ in glob.glob(image_gallery+"/*.jpg"):
#     img_path_list.append(img_)

def extract_features(img_path_list, model1, model2, model3):
    features = []
    labels = []
    for img_ in img_path_list:
        print("img_",img_)   
        weights = []     
        with torch.no_grad():
            img = Image.open(img_).convert('RGB')
            # frame = cv2.imread(img)
            w,h = img.size
            ar = w/h
            pid = [(int(Path(img_).stem.split("_")[0])),]
            camid = [(int(Path(img_).stem.split("_")[1][1:])),]
            process_img = val_transforms_1(img)
            process_img=process_img[None,]

            process_img = process_img.to(device)
            feat1 = model1(x=process_img, cam_label=camid, view_label=0)
            process_img = val_transforms_2(img)
            process_img=process_img[None,]

            process_img = process_img.to(device)
            feat2 = model2(x=process_img, cam_label=camid, view_label=0)
            process_img = val_transforms_3(img)
            process_img=process_img[None,]

            process_img = process_img.to(device)
            feat3 = model3(x=process_img, cam_label=camid, view_label=0)
            model_1_ar = 1
            model_2_ar = 212/224
            model_3_ar = 298/224
            if abs(model_1_ar -ar) <= 0.3:
                weight1 = 1.3
            elif abs(model_1_ar -ar)>0.3 and abs(model_1_ar -ar)<=0.6:
                weight1 = 1.0
            else:
                weight1 = 0.9

            if abs(model_2_ar -ar) <= 0.3:
                weight2 = 1.2
            elif abs(model_2_ar -ar)>0.3 and abs(model_2_ar -ar)<=0.6:
                weight2 = 1.0
            else:
                weight2 = 0.9

            if abs(model_3_ar -ar) <= 0.3:
                weight3 = 1.2
            elif abs(model_3_ar -ar)>0.3 and abs(model_3_ar -ar)<=0.6:
                weight3 = 1.0
            else:
                weight3 = 0.9
            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)
            feat = weights[0]*feat1.cpu()+weights[1]*feat2.cpu()+weights[2]*feat3.cpu()
            # print("feat",feat.cpu())
            features.append(feat)
            labels.append(pid)
    return torch.cat(features),labels

features,labels = extract_features(img_path_list, model1, model2, model3)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_results = tsne.fit_transform(features)
from sklearn.svm import SVC
import numpy as np

classifier = SVC(kernel='linear', C=1.0)
classifier.fit(tsne_results, labels)
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

X0, X1 = tsne_results[:, 0], tsne_results[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots()
plot_contours(ax, classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('TSNE-2')
ax.set_xlabel('TSNE-1')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('Decision Boundary with TSNE Reduced Features',fontweight='bold')
plt.savefig(name_+"Decision Boundary with TSNE Reduced Features.png")
plt.show()