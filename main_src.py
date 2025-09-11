#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:57:37 2025

@author: waqar
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import csv
import random
from PIL import Image
import os
from Model_src import SiameseMobileNetV2
from Train_src import train
from Evaluate_src import evaluate
from utils_src import *


device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)

train_images_path = ".../path_train_images/"
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
size = (IMAGE_HEIGHT, IMAGE_WIDTH)

class CustomDataset_test(Dataset):
    def __init__(self, data, path, transform=None):
        self.csv_data = data
        self.path = path
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        img1_name = self.csv_data["Image1"][idx]
        img2_name = self.csv_data["Image2"][idx]
        img1 = Image.open(train_images_path + img1_name)
        img2 = Image.open(train_images_path + img2_name)
        label = self.csv_data["Label"][idx]
        
        # Apply image transformations
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label, img1_name, img2_name



# Custom transformation functions
def zoom_image(image):
    """Randomly zooms into the image."""
    zoom_factor = random.uniform(1.4, 1.6)
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    y1, x1 = (h - new_h) // 2, (w - new_w) // 2
    zoomed = image[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(zoomed, (w, h))

def random_crop(image):
    """Randomly crops a portion of the image and resizes it back to original dimensions."""
    h, w = image.shape[:2]
    crop_size = random.uniform(0.4, 0.6)
    new_h, new_w = int(h * crop_size), int(w * crop_size)
    y1, x1 = random.randint(0, h - new_h), random.randint(0, w - new_w)
    cropped = image[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h))

def mirror_image(image):
    """Randomly mirrors (flips) the image horizontally."""
    return cv2.flip(image, 1)


class CustomDataset(Dataset):
    def __init__(self, csv_data_path, folder1_path, transform=None, batch_size=32):
        # Load CSV data for folder1
        self.csv_data = csv_data_path
        self.folder1_path = folder1_path
        self.transform = transform
        self.batch_size = batch_size

        # Initialize labels
        self.labels = self.initialize_labels()
        


    def initialize_labels(self):
        """Initializes the labels attribute for easy updating later on."""
        labels = []
        
        # Load labels from CSV data for folder1 images
        for idx in range(len(self.csv_data)):
            labels.append(self.csv_data.iloc[idx]["Label"])
        
        return labels

    def random_custom_transform(self, img):
        img_np = np.array(img)
        transform_choice = random.choice([zoom_image, random_crop, mirror_image])
        transformed_img = transform_choice(img_np)
        return Image.fromarray(transformed_img)
    
    def get_class_distribution(self):
        """Returns the count of each class in the dataset."""
        class_counts = Counter(self.labels)
        print(class_counts)
        return dict(class_counts)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        if idx >= len(self):
            return None
#             raise IndexError(f"Index {idx} is out of range for dataset of length {len(self)}")

        

        img1_name = self.csv_data.iloc[idx]["Image1"]
        img2_name = self.csv_data.iloc[idx]["Image2"]
        img1 = Image.open(os.path.join(self.folder1_path, img1_name))
        img2 = Image.open(os.path.join(self.folder1_path, img2_name))
        label = self.labels[idx]  # Retrieve from labels list


        # Apply transformations if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label, img1_name, img2_name
    
    

versions = ['Re_Id_train_Beta_SOD']

runs = [1]
loss_types = ['All']
for loss_type in loss_types:
    for run in runs:
        for v in versions:
            batch_size= 32

            total_noise_levels = [10,20,30]
            
            
            for noise_lev in total_noise_levels:
                ones_counts = 0
                data_csv_test = list()
                removed_examples_list = list()
                noise_lev = int(noise_lev)
                
                # mean = [0.36440783, 0.36087347, 0.35251117] # humans
                # std = [0.24694804, 0.2500616, 0.25093601]
                
                print(f"Processing For Noise Level {noise_lev/100}:\n")
                transform_train =transforms.Compose([transforms.Resize(size),
                                                       transforms.ToTensor(),
                                                      transforms.RandomHorizontalFlip(p=0.5),
                                                       transforms.Normalize((0.36440783, 0.36087347, 0.35251117),
                                                                            (0.24694804, 0.2500616, 0.25093601))
                                                     ])
                transform_test = transforms.Compose([transforms.Resize(size),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.36440783, 0.36087347, 0.35251117),
                                                                            (0.24694804, 0.2500616, 0.25093601))
                                                     ])
                # model = DNN()
                model = SiameseMobileNetV2()
            
                model = model.to(device)
                LR = 0.0001
            
                optimizer = optim.Adam(model.parameters(),lr=LR, betas=(0.9, 0.999), eps=1e-8)
                criterion = nn.CrossEntropyLoss()

                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                               step_size=7,
                                                               gamma=0.1)
            
                model_info = 'humans_' + str(int(noise_lev)) + '_' + str(int((noise_lev * 2))) + '_filtering_Both_'
            
                folder = str(int(noise_lev)) + '_' + str(int((noise_lev * 2)))  # '20_40'

                base_folder = 'path_to_base_folder' + v + '/' + str(run) + '/' + folder + '/'
                
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder) 
                
                
                file = pd.read_csv('path_to_pairs_csv_file.csv')
                print('Loaded file is: humans_train_pairs_' + str(noise_lev) + '_noisy.csv')

                file_test = pd.read_csv('path_to_test_file.csv')
                # print(file)
                label = file['Label'].to_numpy()
                # print(label)
                unique, counts = np.unique(label, return_counts = True)
                
                print('Train pairs size: ',len(file))
                print('Test pairs size: ', len(file_test))
                
                train_dataset = CustomDataset(file, train_images_path, transform=transform_train)
                
                test_dataset = CustomDataset_test(file_test, train_images_path, transform=transform_test)
                
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                
                
                
                temp = np.inf
                temp_acc = 0.0
                epochs = 30
                

                initial_dataset_size = len(train_dataloader.dataset)  # Assuming train_dataloader.dataset holds the dataset
                class_distribution = train_dataloader.dataset.get_class_distribution()

                    # print('here')
                removed_count = 0  # Track the total number of removed examples
                alpha_c, beta_c = 20, 1.5
                alpha11, beta11 = 1.5, 20
                for epoch in range(epochs):
                        print('Epoch No: ', epoch)
                        loss, train_dataloader, removed_count,train_stop, \
                        alpha_c, beta_c, alpha11, \
                        beta11 = train(base_folder, folder,v,model,epoch, train_dataloader,
                                                                      removed_count,
                                                                      (noise_lev / 100),
                                                                      data_csv_test,
                                                                      removed_examples_list,
                                                                      ones_counts,run,alpha_c,
                                                                      beta_c,alpha11, beta11,
                                                                      loss_type = loss_type)
                        
                        
                        valid_acc, valid_loss = evaluate(base_folder, epoch, model, test_dataloader, criterion,
                                                            loss_type=loss_type)
                        lr_scheduler.step()
                        
                        print(f'[{epoch:02d}] train loss: {loss:0.04f}  ',
                                f'valid loss: {valid_loss:0.04f}  ',
                                f'Valid acc: {valid_acc:0.04f}'
                        )
                        model_path = 'path_to_save_model' + v + '/' + str(run) + '/Models/'
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        
                        if valid_acc > temp_acc:
                            temp_acc = valid_acc
                            torch.save(
                                model, model_path + model_info + mod_type +
                                f'_Valid_Acc_{valid_acc:0.3f}.pkl')
                            print(f'Model saved to {model_path}')
