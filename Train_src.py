#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:56:30 2025

@author: waqar
"""


import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import csv
import os
import torch

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)


#### EM Algortihm using Beta distribution
from EM_algorithm import *
from EM_algorithm_whole import *

def train(base_folder, folder,v, model, epoch,train_dataloader,total_to_remove, 
          removed_count, total_noise,data_csv_test,removed_examples_list,modified_examples,
          ones_counts,run,alpha_c, beta_c, alpha11, beta11,
          loss_type = 'All'):


    print('Loss Type: ', loss_type)
    removal_indices = set()
    status = False

    removed_examples = []  # To store examples being removed in this epoch
    removed_examples_0 = []
    total_losses = []
    weights = []
    similar_distances = []
    similar_distances_0 = []
    remaining_examples = []
    remaining_examples_0 = []
    similar_distances_whole = []
    zeross = []
    loss_const = 0.45
    train_stop = False
    


    for batch_idx, data in enumerate(tqdm(train_dataloader, 0)):
        if data is None:
            print("Data None...")
            continue
        
        img1, img2, label, img1_name, img2_name = data
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        # Skip batch if all indices are marked for removal
        batch_start = batch_idx * len(label)
        if all(idx in removal_indices for idx in range(batch_start, batch_start + len(label))):
            continue

        # Update model parameters
        optimizer.zero_grad()
        out, cosine_loss, contrastive_loss, pairwisee_loss,eu_dist, outt1, outt2 = model(img1, img2, label)
#         print(cosine_loss)
        

        probabilities = F.softmax(out, dim=1)

        prob_cpu = probabilities.detach().cpu().numpy()

        tx_prob = np.max(prob_cpu, axis=1)
        


        ce_loss = criterion(out, label)
        weighted_loss = ce_loss

        total_loss = ce_loss + (loss_const * cosine_loss) + ((1 - loss_const) * contrastive_loss.mean())

        total_loss.backward()
        optimizer.step()

        # Log losses
        total_losses.append(total_loss.item())
        weights.append(cosine_loss.item())

        # Collect distances for similar pairs (label == 1)
        if (epoch >= 5) and (epoch % 5 == 0):

            for i in range(len(label)):
                global_idx = batch_start + i  # Unique index per image pair


                similar_distances_whole.append((global_idx, pairwisee_loss[i].item(), img1_name[i], img2_name[i], 0, prob_cpu[i][0], 0))
                if label[i] == 1:
                    similar_distances.append((global_idx, pairwisee_loss[i].item(), img1_name[i], img2_name[i], 0, prob_cpu[i][0], 0))
                    
                if label[i] == 0:
                    zeross.append((global_idx, img1_name[i], img2_name[i]))
                    similar_distances_0.append((global_idx, pairwisee_loss[i].item(), img1_name[i], img2_name[i], 0, prob_cpu[i][0], 0))
                    
                
    if (epoch >= 5) and (epoch % 5 == 0) and len(similar_distances) > 1:

        indices, distances, img1_names, img2_names,_,_,_ = zip(*similar_distances)

        noisy_examples = [(idx, dist, img1, img2,prob, soft_prob, prob_loss) for idx, dist, img1, img2,prob, soft_prob, prob_loss in similar_distances]
        noisy_examples.sort(key=lambda x: x[1], reverse=False)
        
        noisy_examples_whole = [(idx, dist, img1, img2,prob, soft_prob, prob_loss) for idx, dist, img1, img2, prob, soft_prob, prob_loss in similar_distances_whole]
        
        examples_folder = base_folder +'/examples_rem/' + str(epoch) + '/'
        if not os.path.exists(examples_folder):
            os.makedirs( examples_folder)
        
        csv_all_before = examples_folder +'All_Class1_before_' + str(epoch) + '.csv'
        file_exists_all_before = os.path.isfile(csv_all_before)
        
        with open(csv_all_before, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists_all_before:
                writer.writerow(["idx", "Epoch", "Image1", "Image2", "Distance","FCNN_prob", "Both_prob","prob_loss"])

                
            
            # Write each removed example with epoch, image names, distance, and removal time
            for idx,distance, img1_name, img2_name, prob, soft_prob, prob_loss in noisy_examples:
                writer.writerow([idx,epoch, img1_name, img2_name, distance, prob, soft_prob, prob_loss])
        
        indices_0, distances_0, img1_names_0, img2_names_0,_,_,_ = zip(*similar_distances_0)
        noisy_examples_0 = [(idx, dist, img1, img2, prob, soft_prob,prob_loss) for idx, dist, img1, img2, prob, soft_prob,prob_loss in similar_distances_0]
        noisy_examples_0.sort(key=lambda x: x[1], reverse=True)
        examples_folder = base_folder +'/examples_rem/' + str(epoch) + '/'
        if not os.path.exists(examples_folder):
            os.makedirs( examples_folder)
        
        csv_all_before_0 = examples_folder +'All_Class0_before_' + str(epoch) + '.csv'
        file_exists_all_before_0 = os.path.isfile(csv_all_before_0)
        
        with open(csv_all_before_0, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists_all_before_0:
                writer.writerow(["idx", "Epoch", "Image1", "Image2", "Distance", "FCNN_prob", "Both_prob","prob_loss"])

                
            
            # Write each removed example with epoch, image names, distance, and removal time
            for idx,distance, img1_name, img2_name, prob, soft_prob, prob_loss in noisy_examples_0:
                writer.writerow([idx,epoch, img1_name, img2_name, distance, prob, soft_prob, prob_loss])
                
        #### whole dataset  #######
        # noisy_examples_whole = [(idx, dist, img1, img2, prob, soft_prob,prob_loss) for idx, dist, img1, img2, prob, soft_prob,prob_loss in similar_distances_0]
        csv_all_before_whole = examples_folder +'All_Classes_before_' + str(epoch) + '.csv'
        file_exists_all_before_whole = os.path.isfile(csv_all_before_whole)
        
        with open(csv_all_before_whole, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists_all_before_whole:
                writer.writerow(["idx", "Epoch", "Image1", "Image2", "Distance", "FCNN_prob", "Both_prob","prob_loss"])

                
            
            # Write each removed example with epoch, image names, distance, and removal time
            for idx,distance, img1_name, img2_name, prob, soft_prob, prob_loss in noisy_examples_whole:
                writer.writerow([idx,epoch, img1_name, img2_name, distance, prob, soft_prob, prob_loss])
                
        final_parameters_em = main_em_algorithm_whole(base_folder, epoch,total_noise, v ,noisy_examples_whole, noisy_examples, noisy_examples_0)
        alpha_0, beta_0 = final_parameters_em[3], final_parameters_em[4]
        w1_final,alpha_1, beta_1, itr1 = main_em_algorithm(base_folder, epoch,total_noise, v ,noisy_examples,noisy_examples_0,alpha11,beta11, class_label = 1)

        

        

        alpha_01, beta_01 = final_parameters_em[1], final_parameters_em[2]
        w0_final,alpha_00, beta_00, itr0 = main_em_algorithm(base_folder, epoch,total_noise, v ,noisy_examples_0,alpha_01, beta_01, class_label = 0)
        # print('prob_both 0: ', prob_both[0])

        out_csv = base_folder + str(total_noise) + '_EM_algo.csv'
        out_csv_isfile = os.path.isfile(out_csv)
        
        with open(out_csv, 'a') as csvfile:
            writer = csv.writer(csvfile)
            
            if not out_csv_isfile:
                writer.writerow(['Noise_Level', 'Epoch', 'Alhpa1', 'Theta1','Alpha0','Theta0','W_1','W_0','EM Itr 1', 'EM Itr 0'])
                
            # writer.writerow([total_noise, epoch, alpha_1, beta_1, alpha_00, beta_00, w1_final,w0_final,
                              # itr1, itr0])
        class_distribution = train_dataloader.dataset.get_class_distribution()  # Assuming train_dataloader.dataset holds the dataset
        initial_dataset_size_0 = class_distribution.get(0)
        initial_dataset_size_1 = class_distribution.get(1)
        # train_dataloader_org = len(train_dataloader.dataset)
        if w0_final > 0.003:

            init_dataset_size = initial_dataset_size_0
            num_to_remove_0 = int(init_dataset_size * w0_final)
            # num_to_remove = int(len(noisy_examples) * removal_percentage)
            print('num_to_remove for class 0: ', num_to_remove_0)
            worst_noisy_examples_0 = noisy_examples_0[:num_to_remove_0]
            remains_0 = noisy_examples_0[num_to_remove_0:]
            worst_removed_examples_0 = [(img1, img2) for _,_, img1, img2,_,_,_ in worst_noisy_examples_0]
            removed_examples_0.extend([(idx,dist, img1, img2,prob, soft_prob,proba_loss) for (idx,dist, img1, img2,prob, soft_prob, proba_loss) in (worst_noisy_examples_0)])
            # remaining_examples_0.extend([(idx,dist, img1, img2,prob, soft_prob) for idx,dist, img1, img2,prob, soft_prob in remains_0])
            
            if removed_examples_0:
                log_removed_examples_0(base_folder,folder,v, epoch,removed_examples_0)
                
            train_dataloader = update_dataloader(train_dataloader, zeros_removal = worst_removed_examples_0)
            removed_examples_0.clear()
            
        else:
            train_dataloader = train_dataloader

        if w1_final > 0.003:
            
            # num_to_remove = 80
            
            init_dataset_size = initial_dataset_size_1
            num_to_remove = int(init_dataset_size * w1_final)
            #num_to_remove = int(len(noisy_examples) * removal_percentage)
            print('num_to_remove for class 1: ', num_to_remove)

    
            
            worst_noisy_examples = noisy_examples[:num_to_remove]
            # remains = noisy_examples[num_to_remove:]
            worst_removed_examples = [(img1, img2) for _,_, img1, img2,_,_,_ in worst_noisy_examples]
    
    #         removal_indices.update(global_idx for idx, _, _, _ in worst_noisy_examples)
            removed_examples.extend([(idx,dist, img1, img2,prob, soft_prob, proba_loss) for (idx,dist, img1, img2,prob, soft_prob, proba_loss) in (worst_noisy_examples)])

            if removed_examples:
                log_removed_examples(base_folder, folder,v, epoch, removed_examples,remaining_examples, noisy_examples,
                                      len(worst_noisy_examples), total_noise, noisy_examples_0, data_csv_test,
                                      removed_examples_list,modified_examples,ones_counts,run)
    
            
            # At the end of the epoch:
            train_dataloader = update_dataloader(train_dataloader, removed_examples = worst_removed_examples)
            # train_dataloader = train_dataloader
            
    
                
            removal_indices.clear()
            removed_examples.clear()
        else:
            # continue
            train_dataloader = train_dataloader



    # Calculate the average training loss
    train_loss = np.mean(total_losses)
    return train_loss, train_dataloader, removed_count, train_stop, alpha_c, beta_c, alpha11, beta11


