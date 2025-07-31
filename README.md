# EEG-SCMM: Soft Contrastive Masked Modeling for Cross-Corpus EEG-Based Emotion Recognition (ACMMM 2025)
### Qile Liu (liuqile2022@email.szu.edu.cn), Weishan Ye, Lingli Zhang, and Zhen Liang*.
This repository contains the official implementation of the paper **"EEG-SCMM: Soft Contrastive Masked Modeling for Cross-Corpus EEG-Based Emotion Recognition,"** accepted in ACMMM 2025.
+ [ACMMM 2025](https://dl.acm.org/doi/10.1145/3746027.3755798)
+ [ArXiv Preprint](https://arxiv.org/abs/2408.09186)

# Overview
This paper proposes a novel **S**oft **C**ontrastive **M**asked **M**odeling (**SCMM**) framework to tackle the critical challenge of cross-corpus generalization in the field of EEG-based emotion recognition. Unlike traditional contrastive learning-based models, CLSCMM integrates soft contrastive learning with a hybrid masking strategy to effectively capture the "short-term continuity" characteristics inherent in human emotions and produce stable and generalizable EEG representations. Additionally, a similarity-aware aggregator is introduced to fuse complementary information from semantically related samples, thereby enhancing the fine-grained feature representation capability of the model. Extensive experimental results on three well-known datasets (SEED, SEED-IV, and DEAP) demonstrate that SCMM consistently achieves state-of-the-art (SOTA) performance in cross-corpus EEG-based emotion recognition tasks under both same-class and different-class conditions. In summary, the main contributions of SCMM are outlined as follows:
+ We propose a novel SCMM framework to address three key challenges (insufficient generalization capability, modeling strategy limitation, and ignorance of emotional continuity) in cross-corpus EEG-based emotion recognition.
+ Inspired by the nature of emotions, we introduce a soft weighting mechanism that assigns similarity scores to sample pairs to capture the similarity relationships between different samples. As a result, better feature representations of EEG signals are learned in a self-supervised manner.
+ We develop a new hybrid masking strategy to generate diverse masked samples by considering both channel and feature relationships, which is essential for enhancing contrastive learning. In addition, we introduce a similarity-aware aggregator to fuse complementary information from semantically related samples, enabling fine-grained feature learning and improving the model's overall capability.
+ We conduct extensive experiments on three well-known datasets (SEED, SEED-IV, and DEAP), demonstrating that SCMM achieves SOTA performance against 10 baselines, with an average accuracy improvement of 4.26% under both same-class and different-class cross-corpus settings.

### Key Idea of SCMM
![Key Idea](https://github.com/Kyler-RL/SCMM/blob/main/images/KeyIdea.png)
### Model Structure
![Model Structure](https://github.com/Kyler-RL/SCMM/blob/main/images/ModelStructure.png)
