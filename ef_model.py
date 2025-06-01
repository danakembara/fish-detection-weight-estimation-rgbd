# Import library
import os
import glob
from pathlib import Path
import math
import numpy as np
import pandas as pd
import random
import time
from matplotlib import pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torchvision import models
from PIL import Image
from collections import OrderedDict
from torch.utils.data import DataLoader


# Root folder
root = 'C:/Users/kemba/OneDrive/Desktop/Publication/project/datasets/end2end_match_gt/'

"""# Pre-processing datasets

## Read images and annotations
"""

# Define seed to ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Seed
seed = 42
set_seed(seed)

# Define a function to read images
def read_images(image_dir):
    images = []
    image_paths = pd.Series(list(image_dir.glob('**/*.png'))).sort_values()
    images.extend(image_paths)
    return images

def read_annotations(annotation_dir):
    labels = []
    annotation_paths = pd.Series(list(annotation_dir.glob('**/*.txt'))).sort_values()
    for path in annotation_paths:
        with open(path, 'r') as file:
            content = file.read().strip()
            numbers = content.split()

            # Fish weight is the 6th number (index 5)
            fish_weight = float(numbers[5])

            labels.append(fish_weight)
    return labels

"""## Create custom datasets"""

# Create a paired custom dataset
class PairedFishDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_dataframe, d_dataframe, rgb_transforms=None, d_transforms=None):
        self.rgb_dataframe = rgb_dataframe
        self.d_dataframe = d_dataframe
        self.rgb_transforms = rgb_transforms
        self.d_transforms = d_transforms

    def __len__(self):
        return len(self.rgb_dataframe)

    def __getitem__(self, idx):

        rgb_img = Image.open(self.rgb_dataframe.loc[idx, 'filepath'])
        d_img = Image.open(self.d_dataframe.loc[idx, 'filepath'])
        labels = torch.tensor(self.rgb_dataframe.loc[idx, 'label']).float()

        if self.rgb_transforms is not None:
            rgb_img = self.rgb_transforms(rgb_img)
        if self.d_transforms is not None:
            d_img = self.d_transforms(d_img)

        combined_img = torch.cat((rgb_img, d_img), dim=0)

        return combined_img, labels

"""## Create dataframes for all species"""

# Paths to RGB images
test_rgb_dir = Path(root + '/images/images/')

# Paths to depth images
test_d_dir = Path(root + '/depth_images/images/')

# Path to annotations
ann_test_dir = Path(root + '/images/labels/')

# Convert RGB image paths into series
X_rgb_test = pd.Series(read_images(test_rgb_dir), name='filepath')

# Convert depth image paths into series
X_d_test = pd.Series(read_images(test_d_dir), name='filepath')

# Convert label paths into series and the unit from kilogram to gram
y_test = pd.Series(read_annotations(ann_test_dir), name='label') * 1000

# Merge data and create RGB DataFrames
test_rgb_df = pd.concat([X_rgb_test, y_test], axis=1).reset_index(drop=True)

# Merge data and create depth DataFrames
test_d_df = pd.concat([X_d_test, y_test], axis=1).reset_index(drop=True)

"""## Create dataframes for per species"""

# Paths to RGB images per species
img_test_rgb_species_dir = Path(root + '/images/per_species')

# Paths to depth images per species
img_test_d_species_dir = Path(root + '/depth_images/per_species')

# Path to annotations per species
ann_test_species_dir = Path(root + '/images/per_species')

# Create DataFrames per species for RGB images
test_rgb_species_df = []
for img_test_species, ann_test_species in zip(sorted(img_test_rgb_species_dir.glob('*')), sorted(ann_test_species_dir.glob('*'))):
    X_test_species = pd.Series(read_images(img_test_species), name='filepath')
    y_test_species = pd.Series(read_annotations(ann_test_species), name='label') * 1000
    dataframe = pd.concat([X_test_species, y_test_species], axis=1).reset_index(drop=True)
    test_rgb_species_df.append(dataframe)

# Create DataFrames per species for Depth images
test_d_species_df = []
for img_test_species, ann_test_species in zip(sorted(img_test_d_species_dir.glob('*')), sorted(ann_test_species_dir.glob('*'))):
    X_test_species = pd.Series(read_images(img_test_species), name='filepath')
    y_test_species = pd.Series(read_annotations(ann_test_species), name='label') * 1000
    dataframe = pd.concat([X_test_species, y_test_species], axis=1).reset_index(drop=True)
    test_d_species_df.append(dataframe)

"""## Create dataframes for per occlusion level and species"""

# Path to RGB images per occlusion
img_test_rgb_occ_dir = Path(root + '/images/per_occ_species')

# Path to depth images per occlusion
img_test_d_occ_dir = Path(root + '/depth_images/per_occ_species')

# Path to annotations per occlusion
ann_test_occ_dir = Path(root + '/images/per_occ_species')

# Create RGB DataFrames per species and occlusion level
test_rgb_occ_species_df = []
for img_test_occlusion, ann_test_occlusion in zip(sorted(img_test_rgb_occ_dir.glob('*')), sorted(ann_test_occ_dir.glob('*'))):
    for img_test_occlusion_species, ann_test_occlusion_species in zip(sorted(img_test_occlusion.glob('*')), sorted(ann_test_occlusion.glob('*'))):
        X_test_occlusion_species = pd.Series(read_images(img_test_occlusion_species), name='filepath')
        y_test_occlusion_species = pd.Series(read_annotations(ann_test_occlusion_species), name='label') * 1000
        dataframe = pd.concat([X_test_occlusion_species, y_test_occlusion_species], axis=1).reset_index(drop=True)
        test_rgb_occ_species_df.append(dataframe)

# Create depth DataFrames per species and occlusion level
test_d_occ_species_df = []
for img_test_occlusion, ann_test_occlusion in zip(sorted(img_test_d_occ_dir.glob('*')), sorted(ann_test_occ_dir.glob('*'))):
    for img_test_occlusion_species, ann_test_occlusion_species in zip(sorted(img_test_occlusion.glob('*')), sorted(ann_test_occlusion.glob('*'))):
        X_test_occlusion_species = pd.Series(read_images(img_test_occlusion_species), name='filepath')
        y_test_occlusion_species = pd.Series(read_annotations(ann_test_occlusion_species), name='label') * 1000
        dataframe = pd.concat([X_test_occlusion_species, y_test_occlusion_species], axis=1).reset_index(drop=True)
        test_d_occ_species_df.append(dataframe)

"""# Processing datasets

## Transform images
"""

# Resize and normalize RGB images with ImageNet values
rgb_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Resize and normalize depth images with full training set image values
d_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7149], std=[0.0933])
])

"""## Create data loaders"""

def load_data(batch_size, rgb_dataframe, d_dataframe, rgb_transforms, d_transforms, shuffle):
    dataset = PairedFishDataset(rgb_dataframe=rgb_dataframe, d_dataframe=d_dataframe,
                                rgb_transforms=rgb_transforms, d_transforms=d_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=0 )
    return dataloader

# Check the shape of the combined image
data = PairedFishDataset(test_rgb_df, test_d_df, rgb_transforms=rgb_transforms, d_transforms=d_transforms)
combined_img, label = data[6]
print('The combined image has a shape of', combined_img.shape)

"""## Create performance metrics"""

# Compute mean absolute percentage error
def compute_mape(outputs, labels, eps = 1e-8):
    outputs = outputs.float()
    labels = labels.float()
    loss = torch.abs((outputs-labels)/(labels + eps))
    mape = 100 * torch.mean(loss)
    return mape

"""# Developing ef-DenseNet121

Specification:

*   growth_rate = 32
*   block_config = (6, 12, 24, 16)
*   num_init_features = 64

Source:

https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
"""

# Create DenseLayer
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

# Create DenseBlock
class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

# Create Transition layer
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

# Create EFDenseNet
class EFDenseNet(nn.Module):

    # Key changes: output num_classes = 1
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1, memory_efficient=False):
        super(EFDenseNet, self).__init__()

        # Key changes: first Conv2D in_channels = 4
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(4, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Add multiple dense blocks based on configuration
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:

                # Add transition layer between denseblocks to downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer for regression
        self.regressor = nn.Linear(num_features, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # Update the forward method to use the regressor layer
        out = self.regressor(out)
        return out

# Check the model summary
model = EFDenseNet()

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

"""# Test models

## MAE and MAPE all species
"""

# Hyperparameters
batch_size = 32
learning_rate = 0.001

# Load the test dataset, shuffle = False
test_loader = load_data(batch_size,
                        test_rgb_df, test_d_df,
                        rgb_transforms, d_transforms,
                        shuffle=False)

# Load model and weights
load_model = EFDenseNet()
model_path = 'C:/Users/kemba/OneDrive/Desktop/Publication/project/weight_ef_model.pt'
load_model.load_state_dict(torch.load(model_path))

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model.to(device)

# Initialize loss and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(load_model.parameters(), lr=learning_rate)

# # Test loop
load_model.eval()
test_loss = 0.0
test_mape = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = load_model(images).squeeze()

        # Calculate val loss
        loss = criterion(outputs, labels)

        # Calculate MAPE
        mape = compute_mape(outputs, labels)

        # Sum loss over the batch size
        test_loss += loss.item() * images.size(0)

        # Sum MAPE over the batch size
        test_mape += mape.item() * images.size(0)

    # Calculate average val loss for the epoch
    avg_test_loss = round((test_loss / len(test_loader.dataset)), 2)
    avg_test_mape = round((test_mape / len(test_loader.dataset)), 2)

# Print result
print(f"Test All Species - MAE: {avg_test_loss}, Test MAPE: {avg_test_mape}%")

""" ## MAE and MAPE per species
"""

# Hyperparameters
batch_size = 32
learning_rate = 0.001

# Per species loader
test_species_loader = []
for rgb_df, d_df in zip(test_rgb_species_df, test_d_species_df):
    loader = load_data(batch_size,
                       rgb_df, d_df,
                       rgb_transforms, d_transforms,
                       shuffle=False)
    test_species_loader.append(loader)

# Load model and weights
load_model = EFDenseNet()
model_path = 'C:/Users/kemba/OneDrive/Desktop/Publication/project/weight_ef_model.pt'
load_model.load_state_dict(torch.load(model_path))

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model.to(device)

# Initialize loss and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(load_model.parameters(), lr=learning_rate)

# Initialize species index
species_index = 0

# Test loop
for loader in test_species_loader:
    load_model.eval()
    test_loss = 0.0
    test_mape = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = load_model(images).squeeze()

            # Calculate val loss
            loss = criterion(outputs, labels)

            # Calculate MAPE
            mape = compute_mape(outputs, labels)

            # Sum loss over the batch size
            test_loss += loss.item() * images.size(0)

            # Sum MAPE over the batch size
            test_mape += mape.item() * images.size(0)

    # Calculate average val loss for the epoch
    avg_test_loss = round((test_loss / len(loader.dataset)), 2)
    avg_test_mape = round((test_mape / len(loader.dataset)), 2)

    # Print result
    print(f"Test Species {species_index} - MAE: {avg_test_loss}, MAPE: {avg_test_mape}%")

    # Increment species index
    species_index += 1

""" ## MAE and MAPE per occ and species
"""

# Hyperparameters
batch_size = 32
learning_rate = 0.001

# Per occlusion loader
test_occlusion_loader = []
for rgb_df, d_df in zip(test_rgb_occ_species_df, test_d_occ_species_df):
    loader = load_data(batch_size,
                       rgb_df, d_df,
                       rgb_transforms, d_transforms,
                       shuffle=False)
    test_occlusion_loader.append(loader)

# Load model and weights
load_model = EFDenseNet()
model_path = 'C:/Users/kemba/OneDrive/Desktop/Publication/project/weight_ef_model.pt'
load_model.load_state_dict(torch.load(model_path))

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model.to(device)

# Initialize loss and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(load_model.parameters(), lr=learning_rate)

# Initialize occlusion index
occlusion_index = 0

# Test loop
for loader in test_occlusion_loader:
    load_model.eval()
    test_loss = 0.0
    test_mape = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = load_model(images).squeeze()

            # Calculate val loss
            loss = criterion(outputs, labels)

            # Calculate MAPE
            mape = compute_mape(outputs, labels)

            # Sum loss over the batch size
            test_loss += loss.item() * images.size(0)

            # Sum MAPE over the batch size
            test_mape += mape.item() * images.size(0)

    # Calculate average val loss for the epoch
    avg_test_loss = round((test_loss / len(loader.dataset)), 2)
    avg_test_mape = round((test_mape / len(loader.dataset)), 2)

    # Print result
    print(f"Test Occlusion {occlusion_index} - MAE: {avg_test_loss}, MAPE: {avg_test_mape}%")

    # Increment species index
    occlusion_index += 1