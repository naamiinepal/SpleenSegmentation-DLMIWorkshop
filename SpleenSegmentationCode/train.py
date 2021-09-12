import os
from glob import glob

import ignite
import numpy as np
import torch
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import DiceCoefficient, Loss
from ignite.metrics.accuracy import Accuracy
from ignite.utils import setup_logger
from monai import engines
from monai.apps.utils import download_and_extract
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import from_engine, stats_handler
from monai.handlers.checkpoint_loader import Checkpoint
from monai.handlers.checkpoint_saver import CheckpointSaver
from monai.handlers.stats_handler import StatsHandler
from monai.inferers.inferer import SimpleInferer
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import ActivationsD
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (CropForegroundD,
                                                 RandCropByPosNegLabelD)
from monai.transforms.intensity.dictionary import ScaleIntensityRangeD
from monai.transforms.io.dictionary import LoadImageD
from monai.transforms.post.array import KeepLargestConnectedComponent
from monai.transforms.post.dictionary import (Activationsd, AsDiscreteD,
                                              AsDiscreted,
                                              KeepLargestConnectedComponentd)
from monai.transforms.spatial.dictionary import OrientationD, SpacingD
from monai.transforms.utility.dictionary import (CastToTypeD,
                                                 EnsureChannelFirstD,
                                                 SqueezeDimD, ToTensorD)
from monai.utils.enums import Activation, CommonKeys
from monai.utils.misc import first, set_determinism
from numpy.core import numeric
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import BCEWithLogitsLoss, NLLLoss
from torch.optim import Adam

from utils import slidingWindowEvaluator
from utils.slidingWindowEvaluator import create_sliding_window_evaluator

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
    # CastToTypeD(keys=keys[1], dtype=torch.long),  # conversion of labels to long from float to keep CrossEntropyLoss() from complaining
    # SqueezeDimD(keys=keys[1], dim=0)  # remove channel dim for label to be compatible with crossentropyloss

])
batch_size = 4
spleen_train_dataset = Dataset(training_dicts, spleen_transforms)
spleen_val_dataset = Dataset(validation_dicts, spleen_transforms)
spleen_train_dataloader = DataLoader(spleen_train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=4, drop_last=True)
spleen_val_dataloader = DataLoader(spleen_val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=4, drop_last=True)

# Define network and optimizer
learning_rate = 1e-3
device = torch.device("cuda:0")
channels = (2, 8, 16, 32, 64)
strides = (1, 1, 2, 2, 2)

net = UNet(dimensions=3, in_channels=1, out_channels=1,
           channels=channels, strides=strides, num_res_units=0).to(device)
loss_function = BCEWithLogitsLoss()
optimizer = Adam(net.parameters(), learning_rate)

# test the workflow

# batch = first(spleen_train_dataloader)
# print('Image label shape', batch[keys[0]].shape, batch[keys[1]].shape, )
# target = batch[keys[1]].to(device)
# out = net(batch[keys[0]].to(device))
# max_val = torch.max(out)
# print('Output shape', out.shape, 'Max val', max_val.item())
# loss_val = loss_function(out, target)
# print(f'Loss val {loss_val.item():.3f}')
# print(net)


# train network

num_epoch = 20
best_metric = -1
best_metric_epoch = -1
step = 1
iter_loss_values = list()
batch_sizes = list()
train_loss_values = list()
validation_loss_values = list()


def prep_batch(batch, device, non_blocking):
    return batch[CommonKeys.IMAGE].to(device), batch[CommonKeys.LABEL].to(device)


val_metrics = {
    "nll": Loss(loss_function, device=device)
}

evaluator = create_supervised_evaluator(net, val_metrics, device, True, prepare_batch=prep_batch)
trainer = create_supervised_trainer(net, optimizer, loss_function, device, False, prepare_batch=prep_batch)
trainer.logger = setup_logger("Trainer")
evaluator.logger = setup_logger("Evaluator")


@trainer.on(Events.ITERATION_COMPLETED)
def log_iteration_loss(engine: Engine):
    global step
    loss = engine.state.output
    iter_loss_values.append(loss)
    print(f'epoch {engine.state.epoch}/{engine.state.max_epochs} step {step}/{len(spleen_train_dataloader)} training_loss {loss:.3f}')
    step += 1


@trainer.on(Events.EPOCH_COMPLETED)
def run_validation(engine: Engine):
    global step

    evaluator.run(spleen_val_dataloader)

    # fetch and report validation metrics
    val_nll_loss = evaluator.state.metrics['nll']
    validation_loss_values.append(val_nll_loss)

    # fetch and report training loss
    evaluator.run(spleen_train_dataloader)
    train_nll_loss = evaluator.state.metrics['nll']
    train_loss_values.append(train_nll_loss)
    print(f'epoch {engine.state.epoch}/{engine.state.max_epochs} Train loss {train_nll_loss:.3f} Validation loss {val_nll_loss:.3f}')
    step = 1


# create a model checkpoint to save the network
def _score(_):
    '''
    we used a loss function, so a negative sign is attached to convert it into a metric
    '''
    return -validation_loss_values[-1]


checkpoint_handler = ModelCheckpoint('./TrainedModels', filename_prefix='Unet_nll', score_name='neg_val_loss',
                                     n_saved=1, require_empty=False, score_function=_score)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={'model': net})

trainer.run(spleen_train_dataloader, num_epoch)

best_val_nll_score = min(validation_loss_values)
print(f'Train complete. best_metric {best_val_nll_score:.3f} at epoch {validation_loss_values.index(best_val_nll_score)}')
