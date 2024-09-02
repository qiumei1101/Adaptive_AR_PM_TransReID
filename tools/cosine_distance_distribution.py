
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
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 12
torch.cuda.empty_cache()

veri_query_folder ='/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VehicleID_V1.0/small_veh'
veri_gallery_folder ='/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VehicleID_V1.0/small_veh'
veheicleID_query_folder =''
vehicleID_gallery_folder =''
# top_k = 10
#check all the images in query
positive_pair_cosine_dist = []
negative_pair_cosine_dist = []
# device = "cuda:5"

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# dataloader = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/Sec_reid_weaving/image_test/'
#load model and define dataset
# Now you can analyze distances within and between classes
# define distance
def calculate_centroid_distances(class_tokens, labels):
    unique_labels = np.unique(labels.numpy())
    centroids = {label: np.mean(class_tokens.numpy()[labels.numpy() == label], axis=0) for label in unique_labels}
    
    centroid_distances = euclidean_distances(list(centroids.values()))
    
    return centroid_distances, centroids


cfg_path = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VehicleID/deit_transreid_stride.yml'
weight_path = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/vehicleID_vit_transreid_stride_384x384_patch_mixup_in_pixel/transformer_120.pth'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    # parser.add_argument('--veri_query_folder', type=str, default=None,
    #                     help='Input image path')
    parser.add_argument('--veri_query_folder', type=str, default=veri_query_folder,
                        help='Input image folder path')
    parser.add_argument('--veri_gallery_folder', type=str, default=veri_gallery_folder,
                        help='Input image folder path')
   
    parser.add_argument(
        "--config_file", default=cfg_path, help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args
if __name__ == '__main__':
    args = get_args()
    # model = torch.hub.load('facebookresearch/deit:main', 
    #     'deit_tiny_patch16_224', pretrained=True)
    # model.eval()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(weight_path)
    model.eval()
    if args.use_cuda:
        
        if torch.cuda.is_available():
            # device = torch.device("cuda")
            device =torch.cuda.set_device(5)
           
            model = model.cuda()
    transform = transforms.Compose([
        transforms.Resize(cfg.INPUT.SIZE_TRAIN),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # if args.image_path is not None:       
    #     img = Image.open(args.image_path)
    #     img = img.resize(cfg.INPUT.SIZE_TRAIN)
    #     input_tensor = transform(img).unsqueeze(0)

   
    aspect_ratio_diff = []
    class_tokens = []
    labels = []
    print("args.veri_query_folder",args.veri_query_folder)
    index = 0
    for img in glob.glob(args.veri_query_folder+"/*.jpg"):
        print("img",img)    
        lbls = int(Path(img).stem.split("_")[1])
        img = Image.open(img)
        img = img.resize(cfg.INPUT.SIZE_TRAIN)
        input_tensor = transform(img).unsqueeze(0)
        # index_1=0
        # if index<=100:
        for img2 in glob.glob(args.veri_gallery_folder+"/*.jpg"):
                with torch.no_grad():
                    
                    lbls2 = int(Path(img2).stem.split("_")[1])
                    img2 = Image.open(img2)
                    img2 = img2.resize(cfg.INPUT.SIZE_TRAIN)
                    input_tensor2 = transform(img2).unsqueeze(0)
                    if img != img2:
                        if args.use_cuda:
                            input_tensor = input_tensor.cuda()
                            outputs1 = model(input_tensor)  # assuming the model outputs the embeddings
                            # print("outputs",outputs.size())
                            # class_tokens.append(outputs)  # assuming class token is at position 0
                            # labels.append(lbls)
                            input_tensor2 = input_tensor2.cuda()
                            outputs2 = model(input_tensor2)  # assuming the model outputs the embeddings
                            # print("outputs",outputs.size())
                            # class_tokens.append(outputs)  # assuming class token is at position 0
                            # labels.append(lbls)
                            # print("outputs",outputs1,outputs2)
                            # print("spatial.distance.cosine(outputs1.cpu(), outputs2.cpu())",spatial.distance.cosine(outputs1.cpu(), outputs2.cpu()))
                            if lbls==lbls2:
                                positive_pair_cosine_dist.append(1-spatial.distance.cosine(outputs1.cpu(), outputs2.cpu()))   
                            else:
                                negative_pair_cosine_dist.append(1-spatial.distance.cosine(outputs1.cpu(), outputs2.cpu()))
            #             index_1+=1
        #             if index_1>1000:
        #                 break

        #     index +=1
        # else:
        #     break
        

        

    plt.figure(figsize=(8,6))
    plt.hist(positive_pair_cosine_dist, bins=64, alpha=0.5, label="data1")
    plt.hist(negative_pair_cosine_dist, bins=64, alpha=0.5, label="data2")
    # plt.hist(data3, bins=100, alpha=0.5, label="data3")
    plt.xlabel("Cosine Similarity")
    plt.xlim([-0.5,1])
    plt.ylabel("Count")
    # plt.title("Cosine distance of positive query-gallery pair", fonsize = 14, fontweight='bold')
    plt.legend(['Positive query-gallery pair','Negative query-gallery pair'],loc='best')
    plt.savefig("consine_dist_hist_vehicleID_pm_deit_384x384.png")
      
    plt.show()
            
