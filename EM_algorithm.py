#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:11:59 2025

@author: waqar
"""

import numpy as np
from scipy.special import gamma
from scipy.optimize import fmin
from scipy.stats import beta
import os
import csv

def betaNLL(param,*args):
    """
    Negative log likelihood function for beta
    <param>: list for parameters to be fitted.
    <args>: 1-element array containing the sample data.

    Return <nll>: negative log-likelihood to be minimized.
    """

    a, b = param
    data = args[0]
    pdf = beta.pdf(data,a,b,loc=0,scale=1)
    lg = np.log(pdf)
    mask = np.isfinite(lg)
    nll = -lg[mask].sum()
    return nll

def beta_moments(x):
    mean=np.mean(x)
    var=np.var(x,ddof=1)
    alpha1=mean**2*(1-mean)/var-mean
    beta1=alpha1*(1-mean)/mean
    return alpha1, beta1

## alpha and beta using MLE 
def beta_mle(x): 
    result=fmin(betaNLL,[1,1],args=(x,))
    alpha2,beta2=result
    return alpha2, beta2


def beta_pdf_gamma(x, alpha, beta):


    coeff = gamma(alpha + beta) / (gamma(alpha) * gamma(beta))
    return coeff * (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))

def e_step_optimized(data, w0, alpha0, beta0, alpha1, beta1, class_label):

    


    if class_label == 1:
        beta_class0 = beta_pdf_gamma(data, alpha0, beta0)
        beta_class1 = beta_pdf_gamma(data, alpha1, beta1)
        denominator = w0 * beta_class0 + (1 - w0) * beta_class1
        p_class0 = (w0 * beta_class0) / denominator  

        p_class1 = 1 - p_class0
    if class_label == 0:
        beta_class0 = beta_pdf_gamma(data, alpha1, beta1)
        beta_class1 = beta_pdf_gamma(data, alpha0, beta0)
        denominator = w0 * beta_class1 + (1 - w0) * beta_class0
        p_class1 = (w0 * beta_class1) / denominator  

        p_class0 = 1 - p_class1

    return p_class0, p_class1

def m_step(data, p_class0, p_class1, class_label):
    
    # threshold = np.percentile(p_class0, 95)  # Only remove the top 10% of confident class 0 samples
    # class0_samples = data[p_class0 >= threshold]

    if class_label == 1:
        
        class0_samples = data[p_class0 >= 0.5]
        w0_new = (len(class0_samples) / len(data))
        class1_samples = data[p_class1 >= 0.5]    
        alpha1_new, beta1_new = beta_mle(class1_samples)
        
    if class_label == 0:
        class1_samples = data[p_class1 >= 0.5]
        w0_new = (len(class1_samples) / len(data))
        class0_samples = data[p_class0 >= 0.5]
        alpha1_new, beta1_new = beta_mle(class0_samples)
        

    return w0_new, alpha1_new, beta1_new, class0_samples, class1_samples

def em_algorithm(data, w0, alpha0, beta0, alpha1, beta1,class_label, tol=0.001, max_iter=500):

    iteration = 0
    data = np.array(data)
    probability = [0, 0]
    while iteration < max_iter:
        # E-Step
        p_class0, p_class1 = e_step_optimized(data, w0, alpha0, beta0, alpha1, beta1, class_label)

        # M-Step
        w0_new, alpha1_new, beta1_new, class0_samples_final, class1_samples_final = m_step(data, p_class0, p_class1, class_label)

        # Convergence check
        if abs(w0_new - w0) < tol:
            break  # Stop if w0 change is small

        # Update values
        w0, alpha1, beta1 = w0_new, alpha1_new, beta1_new
        iteration += 1
        probability = [p_class0, p_class1]

    return w0, alpha1, beta1, iteration, class0_samples_final, probability


def main_em_algorithm(base_folder,epoch, noise_folder, v, data_class1, alpha0,beta0, class_label = None):
    if class_label == 1:
        alpha_c, beta_c = 20, 1.3
    else:
        alpha_c, beta_c = 20, 1.3
    w0 = 0.02

    data1 = [dist for _, dist, _, _,_,_,_ in data_class1]
    data_class11 = [(img1,img2,dist) for _,dist,img1,img2,_,_,_ in data_class1]

    alpha_0, beta_0 = alpha0, beta0
    w0_final_org, alpha1_final, beta1_final,iteration,class0_final_samples,probability = em_algorithm(data1, w0, 
                                                                  alpha_0, beta_0, alpha_c, beta_c, class_label)
    

    w0_final = w0_final_org
    out_csv = base_folder + str(noise_folder) + '_EM_algo_' + str(class_label) + '.csv'
    out_csv_isfile = os.path.isfile(out_csv)
    
    with open(out_csv, 'a') as csvfile:
        writer = csv.writer(csvfile)
        
        if not out_csv_isfile:
            writer.writerow(['Noise_Level', 'Epoch', 'Alhpa1', 'Beta1','Alpha0','Beta0','W_0','Clipped_W0','EM Itr'])
            
        # writer.writerow([])
        writer.writerow([noise_folder, epoch, alpha1_final, beta1_final,alpha_0,beta_0,
                          w0_final_org,w0_final,iteration])


    return w0_final, alpha1_final, beta1_final, iteration
