# Adaptive_AR_PM_TransReID
<<<<<<< HEAD
''' Assume all the datasets have models trained on 3 different aspect ratios input; '''


'''#for VeRi-776
python dynamically_ar_fusing_veri.py  --cfg1 'path/Adaptive_ar_PM_TransReID/configs/VeRi/vit_transreid_stride_1.yml' --weight1 'path/Adaptive_ar_PM_TransReID/save_weights/veri_vit_transreid_224x224_patch_mixup_in_pixel/transformer_120.pth' --cfg2 'path/Adaptive_ar_PM_TransReID/configs/VeRi/vit_transreid_stride_2.yml' --weight2 'path/Adaptive_ar_PM_TransReID/save_weights/veri_vit_transreid_stride_224x212_patch_mixup_in_pixel/transformer_120.pth' --cfg3 'path/Adaptive_ar_PM_TransReID/configs/VeRi/vit_transreid_stride_3.yml' --weight3 'path/Adaptive_ar_PM_TransReID/save_weights/veri_vit_transreid_stride_224x298_patch_mixup_in_pixel/transformer_120.pth' --num_classes_1 576 --camera_num_1 20 --num_classes_2 576 --camera_num_2 20 --num_classes_3 576 --camera_num_3 20 --view_num 1 --device cuda:3 --num_query 1678 --image_query  'path/Data_VeReID/VeRi/image_query' --image_gallery 'path/Data_VeReID/VeRi/image_test'
'''
'''for vehicleID
python dynamically_ar_choosing_veh.py  --cfg1 'path/Adaptive_ar_PM_TransReID/configs/VehicleID/vit_transreid_stride_1.yml' --weight1 'path/Adaptive_AR_PM_TransReID/save_weights/vehicleID_vit_transreid_stride_384x384_patch_mixup_in_pixel/transformer_120.pth' --cfg2 'path/Adaptive_ar_PM_TransReID/configs/VehicleID/vit_transreid_stride_2.yml' --weight2 'path/Adaptive_ar_PM_TransReID/save_weights/vehicleID_vit_transreid_stride_384x308_patch_mixup_in_pixel/transformer_120.pth' --cfg3 'path/Adaptive_ar_PM_TransReID/configs/VehicleID/vit_transreid_stride_3.yml' --weight3 'path/Adaptive_ar_PM_TransReID/save_weights/vehicleID_vit_transreid_stride_384x396_patch_mixup_in_pixel/transformer_120.pth' --num_classes_1 13164 --camera_num_1 1  --num_classes_2 13164 --camera_num_2 1 --num_classes_3 13164 --camera_num_3 1 --view_num 1 --device cuda:3
'''
=======
'''
Assume all the datasets have models trained on 3 different aspect ratios input;
'''
```#for VeRi-776  
python dynamically_ar_fusing_veri.py  --cfg1 'path/Adaptive_ar_PM_TransReID/configs/VeRi/vit_transreid_stride_1.yml' --weight1 'path/Adaptive_ar_PM_TransReID/save_weights/veri_vit_transreid_224x224_patch_mixup_in_pixel/transformer_120.pth' --cfg2 'path/Adaptive_ar_PM_TransReID/configs/VeRi/vit_transreid_stride_2.yml' --weight2 'path/Adaptive_ar_PM_TransReID/save_weights/veri_vit_transreid_stride_224x212_patch_mixup_in_pixel/transformer_120.pth' --cfg3 'path/Adaptive_ar_PM_TransReID/configs/VeRi/vit_transreid_stride_3.yml' --weight3 'path/Adaptive_ar_PM_TransReID/save_weights/veri_vit_transreid_stride_224x298_patch_mixup_in_pixel/transformer_120.pth' --num_classes_1 576 --camera_num_1 20 --num_classes_2 576 --camera_num_2 20 --num_classes_3 576 --camera_num_3 20 --view_num 1 --device cuda:3 --num_query 1678 --image_query  'path/Data_VeReID/VeRi/image_query' --image_gallery 'path/Data_VeReID/VeRi/image_test'
```
```for vehicleID
python dynamically_ar_choosing_veh.py  --cfg1 'path/Adaptive_ar_PM_TransReID/configs/VehicleID/vit_transreid_stride_1.yml' --weight1 'path/Adaptive_AR_PM_TransReID/save_weights/vehicleID_vit_transreid_stride_384x384_patch_mixup_in_pixel/transformer_120.pth' --cfg2 'path/Adaptive_ar_PM_TransReID/configs/VehicleID/vit_transreid_stride_2.yml' --weight2 'path/Adaptive_ar_PM_TransReID/save_weights/vehicleID_vit_transreid_stride_384x308_patch_mixup_in_pixel/transformer_120.pth' --cfg3 'path/Adaptive_ar_PM_TransReID/configs/VehicleID/vit_transreid_stride_3.yml' --weight3 'path/Adaptive_ar_PM_TransReID/save_weights/vehicleID_vit_transreid_stride_384x396_patch_mixup_in_pixel/transformer_120.pth' --num_classes_1 13164 --camera_num_1 1  --num_classes_2 13164 --camera_num_2 1 --num_classes_3 13164 --camera_num_3 1 --view_num 1 --device cuda:3
```
>>>>>>> 25533c79dac170ea4ce8467b0833f82a19482733
