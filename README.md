# Similarity-based Outlier Detection for Noisy Object Re-Identification Using Beta Mixtures
The paper can be found [here](https://arxiv.org/abs/2509.08926)

### Benchmark Datasets for Evaluation:
The proposed methodology is evaluated on three multi-view Re-ID datasets: [CUHK03](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)  and [Market1501](https://github.com/niruhan/market1501)  for person Re-ID, and [VeRI-776](https://github.com/JDAI-CV/VeRidataset) for cars Re-ID.

## Abstract
Object re-identification (Re-ID) methods are highly sensitive to label noise, which typically leads to significant performance degradation. We address this challenge by reframing Re-ID as a supervised image similarity task and adopting a Siamese network architecture trained to capture discriminative pairwise relationships. Central to our approach is a novel statistical outlier detection (OD) framework, termed Beta-SOD (Beta mixture Similarity-based Outlier Detection), which models the distribution of cosine similarities between embedding pairs using a two-component Beta distribution mixture model.  We establish a novel identifiability result for mixtures of two Beta distributions, ensuring that our learning task is well-posed. The proposed OD step complements the Re-ID architecture combining binary cross-entropy, contrastive, and cosine embedding losses that jointly optimize feature-level similarity learning. We demonstrate the effectiveness of Beta-SOD in de-noising and Re-ID tasks for person Re-ID, on CUHK03 and Market-1501 datasets, and vehicle Re-ID, on VeRi-776 dataset. Our method shows superior performance compared to the state-of-the-art methods across various noise levels (10-30\%), demonstrating both robustness and broad applicability in noisy Re-ID scenarios.


## Proposed Method:
![overall_propose_ABCD](https://github.com/user-attachments/assets/92d724d5-7451-4a86-94db-f7de9e8d3a86)



## Comparison with Person Re-Identification State-of-the-art Methods on Random Noise:
<img width="1257" height="715" alt="image" src="https://github.com/user-attachments/assets/cab54f5a-c2cc-46f0-99bb-ef69eea2ef01" />


## Comparison with Person Re-Identification State-of-the-art Methods on Pattern Noise:
<img width="1228" height="327" alt="image" src="https://github.com/user-attachments/assets/1703f8ac-2319-4d71-80fa-3a7e94fce4cc" />


## Comparison with Vehicle Re-Identification State-of-the-art Methods on Random Noise:
<img width="592" height="501" alt="image" src="https://github.com/user-attachments/assets/b1be4cc2-e43c-4232-8a3f-ae8fe24ab7c9" />


## Ablation Study:
<img width="600" height="232" alt="image" src="https://github.com/user-attachments/assets/2e0948a2-6a96-4d61-be4d-553ab389c064" />



## Conclusion:
We propose a robust object Re-ID approach that integrates a statistical OD framework into training. Our Beta-SOD framework models cosine similarity distributions with a two-component Beta mixture to iteratively filter noisy labels, enabling unsupervised task-specific dataset de-noising. A Siamese network is then trained using combined binary cross-entropy, contrastive, and cosine embedding losses to enhance both classification and feature similarity. Experiments show improved robustness and accuracy across Re-ID benchmarks under varying noise levels. The observed performance gains strongly suggest the effectiveness and generalization capacity of Beta-SOD in real-world noisy label environments.
We present a novel theoretical result demonstrating that the unique identifiability of two-component Beta mixtures is attainable with probability 1. 
Based on the strong performance observed in Re-ID tasks, we expect Beta-mixture modeling to be a versatile and effective tool for de-noising cosine similarity scores produced by Siamese architectures, with potential applications extending to face recognition, speaker verification, and sentence embedding models.
