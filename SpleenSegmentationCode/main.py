import os
from glob import glob

import monai
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from monai.apps.utils import download_and_extract
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.networks.nets import UNet
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (CropForegroundD,
                                                 RandCropByPosNegLabelD)
from monai.transforms.intensity.dictionary import (ScaleIntensityD,
                                                   ScaleIntensityRangeD)
from monai.transforms.io.array import LoadImage
from monai.transforms.io.dictionary import LoadImageD
from monai.transforms.spatial.dictionary import OrientationD, SpacingD
from monai.transforms.utility.dictionary import (AddChannelD,
                                                 EnsureChannelFirstD,
                                                 SqueezeDimD, ToTensorD)
from monai.utils.misc import first, set_determinism
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam

# download and extract spleen segmentation dataset
resource = 'https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar'

root_dir = os.path.join('.', 'Dataset')
compressed_file = os.path.join('.', 'Dataset', "Task09_Spleen.tar")
data_dir = os.path.join('.', 'Dataset', 'Task09_Spleen')
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir)

# set deterministic training for reproducibility
set_determinism(seed=80911923)

# read image filenames from the dataset folders
training_img_names = sorted(glob(os.path.join(data_dir,
                                              'imagesTr', 'spleen_**.nii.gz')))
training_label_names = sorted(glob(os.path.join(data_dir,
                                                'labelsTr', 'spleen_**.nii.gz')))
testing_img_names = sorted(glob(os.path.join(data_dir,
                                             'imagesTs', 'spleen_**.nii.gz')))
# print('Training Set ', len(training_img_names), len(training_label_names))
# print('Testing set', len(testing_img_names))

# check orientation and spacing of dataset
# for name in training_img_names + testing_img_names:
#     img: sitk.Image = sitk.ReadImage(name)
#     print(img.GetSize(), img.GetSpacing(),
#           nib.aff2axcodes(np.array(img.GetDirection()).reshape((3, 3))))
# All images were found to be in same orientation.
# spacing varied widely in axial direction from 1.5 to 8.0

# split training/validation
training_img_names, validation_img_names, training_label_names, validation_label_names = train_test_split(training_img_names, training_label_names, train_size=0.8)
# [print(img, label) for img, label in zip(validation_img_names, validation_label_names)]


# Define MONAI transforms, Dataset and Dataloader to preprocess data
training_dicts = [{"image": image_name, "label": label_name}
                  for image_name, label_name in zip(training_img_names, training_label_names)]


validation_dicts = [{"image": image_name, "label": label_name}
                    for image_name, label_name in
                    zip(validation_img_names, validation_label_names)]


keys = ["image", "label"]
soft_tissue_window = (-50, 150)
output_range = (0.0, 1.0)
spatial_size = (96,)*3
spleen_transforms = Compose([
    LoadImageD(keys=keys),
    EnsureChannelFirstD(keys),  # Almost all transforms in MONAI expect CHWD
    # AddChannelD(keys=keys),
    OrientationD(keys=keys, axcodes='RAS'),
    SpacingD(keys, pixdim=(1.0, 1.0, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRangeD(keys=(keys[0],), a_min=soft_tissue_window[0],
                         a_max=soft_tissue_window[1],
                         b_min=0.0,
                         b_max=1.0,
                         clip=True),
    CropForegroundD(keys=keys, source_key=keys[0]),
    RandCropByPosNegLabelD(keys, keys[1], spatial_size, 1, 1, 4, keys[0]),
    ToTensorD(keys=keys),
    SqueezeDimD(keys=keys[1], dim=0)  # remove channel dim for label to be compatible with crossentropyloss

])
batch_size = 2
spleen_train_dataset = Dataset(training_dicts, spleen_transforms)
spleen_val_dataset = Dataset(validation_dicts, spleen_transforms)
spleen_train_dataloader = DataLoader(spleen_train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=4)
spleen_val_dataloader = DataLoader(spleen_val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=4)

# Define network and optimizer
learning_rate = 1e-3
device = torch.device("cuda:0")
channels = (2, 8, 16, 32, 64)
strides = (1, 1, 2, 2, 2)

net = UNet(dimensions=3, in_channels=1, out_channels=2,
           channels=channels, strides=strides, num_res_units=0).to(device)
loss_function = CrossEntropyLoss()
optimizer = Adam(net.parameters(), learning_rate)

batch = first(spleen_train_dataloader)
print('Image label shape', batch[keys[0]].shape, batch[keys[1]].shape, )
out = net(batch[keys[0]].to(device))
print('Output shape', out.shape)
loss_val = loss_function(out, torch.tensor(batch[keys[1]], dtype=torch.long, device=device))
print(f'Loss val {loss_val.item():.3f}')
# print(net)
# train network

num_epoch = 4
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

for epoch in range(num_epoch):
    print("-"*10)
    print(f"epoch {epoch + 1} / {num_epoch}")
    epoch_loss = 0
    step = 1
    steps_per_epoch = len(spleen_train_dataloader)

    # put the network in train mode
    net.train()
    for batch in spleen_train_dataloader:
        # conversion of labels to long from float to keep CrossEntropyLoss() from complaining
        images, labels = batch[keys[0]].to(device), torch.tensor(batch[keys[1]], dtype=torch.long, device=device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(spleen_train_dataloader)} train loss {loss.item():.3f}")
        step += 1
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch+1} average loss {epoch_loss:.3f}")
