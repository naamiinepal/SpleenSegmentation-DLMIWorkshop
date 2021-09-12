from glob import glob
from os.path import join

import monai
import nibabel as nib
import numpy
import numpy as np
import torch
from monai.data import NiftiSaver
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset, Dataset
from monai.data.nifti_writer import write_nifti
from monai.data.utils import to_affine_nd
from monai.inferers.utils import sliding_window_inference
from monai.networks.nets.unet import UNet
from monai.transforms.compose import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRangeD
from monai.transforms.io.array import LoadImage
from monai.transforms.io.dictionary import LoadImageD
from monai.transforms.post.array import (AsDiscrete,
                                         KeepLargestConnectedComponent)
from monai.transforms.spatial.dictionary import OrientationD, SpacingD
from monai.transforms.utility.array import (EnsureChannelFirst, Lambda,
                                            SqueezeDim)
from monai.transforms.utility.dictionary import (EnsureChannelFirstD,
                                                 SqueezeDimD, ToTensorD)
from monai.transforms.utils import allow_missing_keys_mode
from monai.utils import ImageMetaKey
from monai.utils.enums import CommonKeys
from monai.utils.misc import first, set_determinism
from nibabel.orientations import aff2axcodes
from numpy import dtype, ndarray
from skimage import measure
from sklearn.model_selection import train_test_split

# set deterministic training for reproducibility
set_determinism(seed=80911923)

data_dir = join('.', 'Dataset', 'Task09_Spleen')

# read image filenames from the dataset folders
training_img_names = sorted(glob(join(data_dir,
                                      'imagesTr', 'spleen_**.nii.gz')))
training_label_names = sorted(glob(join(data_dir,
                                        'labelsTr', 'spleen_**.nii.gz')))

# split training/validation
training_img_names, validation_img_names, training_label_names, validation_label_names = train_test_split(training_img_names, training_label_names, train_size=0.8)
validation_dicts = [{"image": image_name, "label": label_name}
                    for image_name, label_name in
                    zip(validation_img_names, validation_label_names)]

keys = ["image", "label"]
soft_tissue_window = (-50, 150)
output_range = (0.0, 1.0)

eval_transforms = Compose([
    LoadImageD(keys),
    EnsureChannelFirstD(keys),
    OrientationD(keys, axcodes='RAS'),
    SpacingD(keys, pixdim=(1.0, 1.0, 2.0), mode=('bilinear', 'nearest')),
    ScaleIntensityRangeD(keys=(keys[0],), a_min=soft_tissue_window[0],
                         a_max=soft_tissue_window[1], b_min=0.0, b_max=1.0, clip=True),
    ToTensorD(keys)

])

# create dataset and dataloader
spleenValDataset = CacheDataset(validation_dicts, eval_transforms,
                                cache_rate=1.0,
                                num_workers=4)
spleenValLoader = DataLoader(spleenValDataset, batch_size=1, num_workers=4)

# print(first(spleenValLoader)['image_meta_dict']['filename_or_obj'])

# setup paths
base_path = join('.', 'TrainedModels', 'Unet_nll')
result_path = join(base_path, 'results')
evaluation_path = join(base_path, 'evaluation')
model_path = join(base_path, 'models', 'Unet_nll_model_neg_val_loss=-0.0887.pt')

device = torch.device("cuda:0")
channels = (2, 8, 16, 32, 64)
strides = (1, 1, 2, 2, 2)

model = UNet(dimensions=3, in_channels=1, out_channels=1,
             channels=channels, strides=strides, num_res_units=0).to(device)

checkpoint = torch.load(model_path)
# print(checkpoint.keys())
model.load_state_dict(checkpoint)

model.eval()

sw_batch_size = 4
roi_size = (96,)*3

seg_saver = NiftiSaver(result_path, mode='nearest')


def run():
    post_transform = Compose([
        AsDiscrete(threshold_values=True),
        KeepLargestConnectedComponent(applied_labels=1)
    ])
    # run sliding window inference
    for i, data in enumerate(spleenValLoader):
        image = data[CommonKeys.IMAGE].to(device)
        with torch.no_grad():
            val_out = sliding_window_inference(image, roi_size, sw_batch_size, model)
        val_out = AsDiscrete(threshold_values=True)(val_out)
        seg_saver.save_batch(val_out, data['image_meta_dict'])

        # arr_val_out = val_out.detach().cpu().numpy()
        # arr_val_out = np.squeeze(arr_val_out)
        # print(arr_val_out.shape)
        # largest_cc = np.zeros(shape=arr_val_out.shape, dtype=arr_val_out.dtype)
        # arr_val_out: ndarray = measure.label(arr_val_out, connectivity=3)
        # if arr_val_out.max() != 0:
        #     largest_cc[...] = arr_val_out == (np.argmax(np.bincount(arr_val_out.flat)[1:])+1)
        # print(largest_cc.shape)
        # meta_data = data['image_meta_dict']
        # print(data.keys())
        # print(meta_data.keys())
        # val_out = post_transform(val_out)
        # print(val_out.shape)
        # seg_saver.save_batch(largest_cc, data['image_meta_dict'])


run()
