#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:46:19 2025

@author: waqar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseMobileNetV2(nn.Module):
    def __init__(self):
        super(SiameseMobileNetV2, self).__init__()
        
        # Load pre-trained MobileNet-v2
        self.mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        self.mobilenet_v2.classifier = nn.Identity()
        

        
        # Extract feature size from the last convolutional layer
        self.n_features = 1280  # MobileNet-v2 outputs 1280-dimensional features


        # Add a classification head similar to the ResNet-18 model
        self.fc = nn.Sequential(
            nn.Linear(self.n_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 2)
        )

    def forward(self, x1, x2, label):
        # Pass both images through MobileNet-v2
        out11 = self.mobilenet_v2(x1)
        out12 = self.mobilenet_v2(x2)
        
        # Normalize the feature vectors
        out1 = F.normalize(out11, p=2, dim=1)
        out2 = F.normalize(out12, p=2, dim=1)
        
        # Compute euclidean distance
        euclidean_distance = F.pairwise_distance(out1, out2)
        margin = 1.0
        contrastive_loss = 0.5 * ((label * euclidean_distance.pow(2)) + 
                          ((1 - label) * F.relu(margin - euclidean_distance).pow(2)))
        
        cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.5)
        cosine_label = 2 * label - 1
        # Calculate cosine embedding loss
        cosine_loss = cosine_embedding_loss(out1, out2, cosine_label)
        
        # Calculate absolute difference
        diff = torch.abs(out1 - out2)
        
        # Classification output
        out = self.fc(diff)
        
        # Calculate cosine similarity for each pair
        cos_sim = F.cosine_similarity(out11, out12, dim=1)
        
        return out, cosine_loss, contrastive_loss, cos_sim,euclidean_distance, out1, out2