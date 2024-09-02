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
from datasets.vehicleid import VehicleID
veh = VehicleID()
train, query, gallery = veh.process_split(True)
print("query",len(query))
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 10
torch.cuda.empty_cache()
device = "cuda:3"

cfg1 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VehicleID/vit_transreid_stride.yml'
weight1= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/vehicleID_vit_transreid_stride_384x384_patch_mixup_in_pixel/transformer_120.pth'

cfg2 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VehicleID/vit_transreid_stride_1.yml'
weight2= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/vehicleID_vit_transreid_stride_384x308_patch_mixup_in_pixel/transformer_120.pth'

cfg3 = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VehicleID/vit_transreid_stride_2.yml'
weight3= '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/vehicleID_vit_transreid_stride_384x396_patch_mixup_in_pixel/transformer_120.pth'

name_ ='Veh_fused'
num_classes_1 = 13164
camera_num_1 = 1
num_classes_2 = 13164
camera_num_2 = 1
num_classes_3 = 13164
camera_num_3 = 1
view_num = 1
num_query = len(query)
import numpy as np
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

SIZE_TEST_1 =  [384, 384]
SIZE_TEST_2 =  [384, 308]
SIZE_TEST_3 =  [384, 396]
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
image_query = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VehicleID_V1.0/small_veh'
# image_gallery ='/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/image_test'
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
            pid = [(int(Path(img_).stem.split("_")[1])),]
            camid = [0,]
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
            model_2_ar = 308/384
            model_3_ar = 396/384
            if abs(model_1_ar -ar) <= 0.4:
                weight1 = 1.2
            elif abs(model_1_ar -ar)>0.4 and abs(model_1_ar -ar)<=0.7:
                weight1 = 1.0
            else:
                weight1 = 0.5

            if abs(model_2_ar -ar) <= 0.4:
                weight2 = 1.2
            elif abs(model_2_ar -ar)>0.4 and abs(model_2_ar -ar)<=0.7:
                weight2 = 1.0
            else:
                weight2 = 0.5

            if abs(model_3_ar -ar) <= 0.4:
                weight3 = 1.2
            elif abs(model_3_ar -ar)>0.4 and abs(model_3_ar -ar)<=0.7:
                weight3 = 1.0
            else:
                weight3 = 0.5
            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)
            feat = weights[0]*feat1.cpu()+weights[1]*feat2.cpu()+weights[2]*feat3.cpu()
            # print("feat",feat.cpu())
            features.append(feat)
            labels.append(pid)
    return torch.cat(features),labels

features,labels = extract_features(img_path_list, model1,model2,model3)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_results = tsne.fit_transform(features)
vis_x = tsne_results[:, 0]
vis_y = tsne_results[:, 1]
plt.scatter(vis_x, vis_y, c=labels)
plt.colorbar(ticks=range(len(labels)))
# plt.clim(-0.5, 9.5)
plt.savefig(name_+" Clustering with TSNE Reduced Features.png")

plt.show()


# from sklearn.svm import SVC
# import numpy as np

# classifier = SVC(kernel='linear', C=1.0)
# classifier.fit(tsne_results, labels)
# import matplotlib.pyplot as plt

# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     return xx, yy

# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out

# X0, X1 = tsne_results[:, 0], tsne_results[:, 1]
# xx, yy = make_meshgrid(X0, X1)

# fig, ax = plt.subplots()
# plot_contours(ax, classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X0, X1, c=labels, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_ylabel('TSNE-2')
# ax.set_xlabel('TSNE-1')
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title('Decision Boundary with TSNE Reduced Features',fontweight='bold')
# plt.savefig(name_+"Decision Boundary with TSNE Reduced Features.png")
# plt.show()