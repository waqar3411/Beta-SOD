#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 17:23:02 2025

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

def e_step_optimized(data, w0, alpha0, beta0, alpha1, beta1):

    beta_class0 = beta_pdf_gamma(data, alpha0, beta0)
    beta_class1 = beta_pdf_gamma(data, alpha1, beta1)


    denominator = w0 * beta_class0 + (1 - w0) * beta_class1
    p_class0 = (w0 * beta_class0) / denominator  

    p_class1 = 1 - p_class0  

    return p_class0, p_class1

def m_step(data, p_class0, p_class1):
    
    # threshold = np.percentile(p_class0, 95)  # Only remove the top 10% of confident class 0 samples
    # class0_samples = data[p_class0 >= threshold]

    class0_samples = data[p_class0 >= 0.5]
    w0_new = (len(class0_samples) / len(data))


    class1_samples = data[p_class1 >= 0.5]
    w1_new = (len(class1_samples) / len(data))
    
    alpha1_new, beta1_new = beta_mle(class1_samples)
    alpha0_new, beta0_new = beta_mle(class0_samples)

    return w0_new, alpha1_new, beta1_new,alpha0_new, beta0_new, class1_samples, class0_samples, w1_new

def em_algorithm_whole(data, w0, alpha0, beta0, alpha1, beta1, tol=0.001, max_iter=500):

    iteration = 0
    data = np.array(data)
    w1 = 0
    while iteration < max_iter:
        # E-Step
        p_class0, p_class1 = e_step_optimized(data, w0, alpha0, beta0, alpha1, beta1)

        # M-Step
        w0_new, alpha1_new, beta1_new, alpha0_new, beta0_new,class1_samples_final, class0_samples_final, w1_new = m_step(data, p_class0, p_class1)

        # Convergence check
        if abs(w0_new - w0) < tol:
            break  # Stop if w0 change is small

        # Update values
        w0, alpha1, beta1, alpha0, beta0, w1 = w0_new, alpha1_new, beta1_new, alpha0_new, beta0_new, w1_new
        iteration += 1
        
    final_parameters = [w0, alpha1, beta1,alpha0, beta0, iteration, class1_samples_final, class0_samples_final, w1]

    return final_parameters



def main_em_algorithm_whole(base_folder,epoch, noise_folder, v, data_whole, data_class1, data_class0):

    w0 = 0.5

    data0 = [dist for _, dist, _, _,_,_,_ in data_class0]
    data1 = [dist for _, dist, _, _,_,_,_ in data_class1]
    data = [dist for _, dist, _, _,_,_,_ in data_whole]

    alpha_0, beta_0 = beta_mle(data0)
    alpha_c, beta_c = beta_mle(data1)
    final_parameters_em = em_algorithm_whole(data, w0, alpha_0, beta_0, alpha_c, beta_c)
    
    w0_final = final_parameters_em[0]
    w1_final = final_parameters_em[8]
    out_csv = base_folder + str(noise_folder) + '_EM_algo_whole_data.csv'
    out_csv_isfile = os.path.isfile(out_csv)
    
    with open(out_csv, 'a') as csvfile:
        writer = csv.writer(csvfile)
        
        if not out_csv_isfile:
            writer.writerow(['Noise_Level', 'Epoch', 'Alhpa1', 'Beta1','Alpha0','Beta0','W_0','W_1','EM Itr'])
            
        
        writer.writerow([noise_folder, epoch, final_parameters_em[1], final_parameters_em[2],
                        final_parameters_em[3],final_parameters_em[4], final_parameters_em[0],
                        w1_final,final_parameters_em[5]])
        
    return final_parameters_em
