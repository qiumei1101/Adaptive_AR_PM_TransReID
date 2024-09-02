import argparse
import cv2
import numpy as np
import torch
import glob
from pathlib import Path
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
cfg_path = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/configs/VehicleID/vit_transreid_stride_1.yml'
weight_path = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/save_weights/vehicleID_vit_transreid_stride_384x308_patch_mixup_in_pixel/transformer_120.pth'
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import os
import os.path as osp
# output_dir = cfg.OUTPUT_DIR
# if output_dir and not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# logger = setup_logger("transreid", output_dir, if_train=False)
# logger.info(args)


# logger.info("Running with config:\n{}".format(cfg))

# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    # parser.add_argument(
    #     '--image-path',
    #     type=str,
    #     default=img_pth,
    #     help='Input image path')
    parser.add_argument(
         "--config_file", default=cfg_path, help="path to config file", type=str
    )
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
 
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=24, width=25):
    # print("tensor.size",tensor.size())
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # model = torch.hub.load('facebookresearch/deit:main',
    #                        'deit_tiny_patch16_224', pretrained=True)
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    model.eval()
    # print("model",model)
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.base.blocks[-1].norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)
    # img_pth = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/Sec_reid_weaving/image_query/1339_c13_0.jpg'

    # image_folder = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/weaving_seperate_data/weaving_1'
    count = 0
    save_folder = '/home/meiqiu@ads.iu.edu/Mei_all/TransReID/attention_map_ViT_veh'
    dataset_dir = '/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VehicleID_V1.0'
    img_dir = osp.join(dataset_dir, 'image')
    split_dir = osp.join(dataset_dir, 'train_test_split')
    train_list = osp.join(split_dir, 'train_list.txt')
    test_list = osp.join(split_dir, 'test_list_2400.txt')

    # num_larger_500 = 0
    with open(test_list) as f_train:
        train_data = f_train.readlines()
        for data in train_data:
            name, pid = data.strip().split(' ')
            img_path = osp.join(img_dir, name+'.jpg')
            print("image path",img_path)
    #         frame = cv2.imread(img_path)
    # for img in sorted(glob.glob('/home/meiqiu@ads.iu.edu/Mei_all/Data_VeReID/VeRi/image_test/*.jpg'),key=lambda x: (os.path.splitext(os.path.basename(x))[0])):
            img_name = Path(img_path).stem
    #         print(img_name)
            rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (308,384))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            targets = None

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            cv2.imwrite(os.path.join(save_folder,img_name+'_'+f'{args.method}_cam.jpg'), cam_image)
            count+=1
            if count>400:
                break
            