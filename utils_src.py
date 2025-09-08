#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:45:31 2025

@author: waqar
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import csv
from PIL import Image
import os



def visualize_pair(img1, img2, label):
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(np.transpose(img1.numpy(), (1, 2, 0)))
    axes[1].imshow(np.transpose(img2.numpy(), (1, 2, 0)))
    if label:
        fig.suptitle('Same', y=1)
    else:
        fig.suptitle('Different', y=1)
        
        
        
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def log_removed_examples_0(base_folder,folder,v, epoch,removed_examples_00):
    if not os.path.exists(base_folder +'/examples_rem/'):
        os.mkdir( base_folder +'/examples_rem/')
    
    csv_filename = base_folder +'/examples_rem/' + str(epoch) + '_rem_0.csv'
    # Check if the CSV file exists; if not, write headers
    file_exists = os.path.isfile(csv_filename)
    csv_all = base_folder +'/examples_rem/All_rem_0.csv'
    file_exists_all = os.path.isfile(csv_all)
    
    with open(csv_all, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write headers if file is new
        if not file_exists_all:
            writer.writerow(["idx", "Epoch", "Image1", "Image2", "Distance", "FCNN_prob", "Both_prob", "proba_loss"])
#             writer.writerow(['No. of Identifies Examples: ', len_noisy_examples])
#             writer.writerow(['No. of Filtered Out Examples: ', len_worst_examples])
            
        
        # Write each removed example with epoch, image names, distance, and removal time
        for idx,distance, img1_name, img2_name, prob, soft_prob,proba_loss in removed_examples_00:
            writer.writerow([idx,epoch, img1_name, img2_name, distance, prob, soft_prob,proba_loss])            
    



def log_removed_examples(base_folder,folder,v, epoch, removed_examples,remaining_examples,
                         len_noisy_examples, len_worst_examples,rem_per,noisy_examples_0, data_csv_test,
                         removed_examples_list, modified_examples, ones_counts,run):
    
    rem_per = int(rem_per * 100)

    
    if not os.path.exists(base_folder +'/examples_rem/'):
        os.mkdir( base_folder +'/examples_rem/')
    
    csv_filename = base_folder +'/examples_rem/' + str(epoch) + '_rem.csv'
    # Check if the CSV file exists; if not, write headers
    file_exists = os.path.isfile(csv_filename)
    csv_all = base_folder +'/examples_rem/All_rem.csv'
    file_exists_all = os.path.isfile(csv_all)
    
    with open(csv_all, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write headers if file is new
        if not file_exists_all:
            writer.writerow(["idx", "Epoch", "Image1", "Image2", "Distance","FCNN_prob", "Both_prob", "proba_loss"])
#             writer.writerow(['No. of Identifies Examples: ', len_noisy_examples])
#             writer.writerow(['No. of Filtered Out Examples: ', len_worst_examples])
            
        
        # Write each removed example with epoch, image names, distance, and removal time
        for idx,distance, img1_name, img2_name, prob, soft_prob,proba_loss in removed_examples:
            writer.writerow([idx,epoch, img1_name, img2_name, distance, prob, soft_prob, proba_loss])
    
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write headers if file is new
        if not file_exists:
#             writer.writerow(['No. of Identifies Examples: ', len(len_noisy_examples)])
#             writer.writerow(['No. of Filtered Out Examples: ', len_worst_examples])
            writer.writerow(["idx", "Epoch", "Image1", "Image2", "Distance", "FCNN_prob", "Both_prob", "proba_loss"])
        
        # Write each removed example with epoch, image names, distance, and removal time
        for idx,distance, img1_name, img2_name, prob, soft_prob,proba_loss in removed_examples:
            writer.writerow([idx,epoch, img1_name, img2_name, distance, prob, soft_prob,proba_loss])
    csv_med = base_folder + folder +'_Median.csv'
    file_exists_med = os.path.isfile(csv_med)
    
    

def filter_dataset(dataset, removal_indices):
#     csv_log_path='/media/waqar/data3/Similarity_v2/Resnet18/' + base_folder +'/filter_log.csv'
#     """Filter dataset attributes based on removal indices."""
    # Create a mask of valid indices
    csv_indices = [idx for idx in range(len(dataset.csv_data)) if idx not in removal_indices]
    
    
    folder2_indices = [idx for idx in range(len(dataset.folder2_images)) 
                       if idx + len(dataset.csv_data) not in removal_indices]
    

    
    # Filter all dataset attributes
    dataset.csv_data = dataset.csv_data.iloc[csv_indices].reset_index(drop=False)
    dataset.folder2_images = [dataset.folder2_images[i] for i in folder2_indices]
#     dataset.labels = [dataset.labels[i] for i in valid_indices]
    dataset.labels = [dataset.labels[i] for i in csv_indices + folder2_indices]
    


def filter_dataset_by_names(dataset, removal_pairs):
    """
    Filters dataset attributes based on pairs of image names.
    
    Args:
        dataset: The dataset object containing csv_data, folder2_images, and labels.
        removal_pairs: A list of tuples [(img1_name, img2_name), ...] to be removed.
    """
    # Convert removal_pairs to a set for efficient lookup
    removal_set = set(removal_pairs)

    # Filter csv_data
    csv_data_filtered = []
    labels_filtered = []
    for idx, row in dataset.csv_data.iterrows():
        pair = (row["Image1"], row["Image2"])
        if pair not in removal_set:
            csv_data_filtered.append(row)
            labels_filtered.append(dataset.labels[idx])

    # Update csv_data and labels for folder1
    dataset.csv_data = pd.DataFrame(csv_data_filtered).reset_index(drop=True)
    

#     dataset.folder2_images = folder2_filtered
    dataset.labels = labels_filtered



def update_dataloader(dataloader, zeros_removal = None, removed_examples = None):
    """Update dataloader after removing examples."""
    if removed_examples != None:
        filter_dataset_by_names(dataloader.dataset, removed_examples)
    if zeros_removal != None:
        filter_dataset_by_names(dataloader.dataset, zeros_removal)
    
    
    return DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, shuffle=True)