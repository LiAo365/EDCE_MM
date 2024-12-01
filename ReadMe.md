#  Efficient Dual-Confounding Eliminating for Weakly-supervised Temporal Action Localization

This repository is the official implementation of the paper [Efficient Dual-Confounding Eliminating for Weakly-supervised Temporal Action Localization](https://dl.acm.org/doi/10.1145/3664647.3681571) in MM '24.

## Abstract
Weakly-supervised Temporal Action Localization (WTAL) following a localization-by-classification paradigm has achieved significant results, yet still grapples with confounding arising from ambiguous snippets. Previous works have attempted to distinguish these ambiguous snippets from action snippets without investigating the underlying causes of their formation, thus failing to effectively eliminate the bias on both action-context and action content. In this paper, we revisit WTAL from the perspective of structural causal model to identify the true origins of confounding, and propose an efficient dual-confounding eliminating framework to alleviate these biases. Specifically, we construct a Substituted Confounder Set (SCS) to eliminate the confounding bias on action context by leveraging the modal disparity between RGB and FLOW. Then, a Multi-level Consistency Mining (MCM) method is designed to mitigate the confounding bias on action-content by utilizing the consistency between discriminative snippets and corresponding proposals at both the feature and label levels. Notably, SCS and MCM could be seamlessly integrated into any two-stream models without additional parameters by Expectation-Maximization (EM) algorithm. Extensive experiments on two challenging benchmarks including THUMOS14 and ActivityNet-1.2 demonstrate the superior performance of our method.

## Baseline Models

We choose three baseline models：
 - [CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net)
 - [DELU](https://github.com/MengyuanChen21/ECCV2022-DELU)
 - [DDG-Net](https://github.com/XiaojunTang22/ICCV2023-DDGNet)

We use the pre-extracted features of THUMOS14 and ActivityNet-1.2 provided by CO2-Net. Just Following the readme file in CO2-Net to download all the pre-extracted features to local machine.

We provide train scripts in three folders, just following the readme file.

## How to use

Our method is designed to be plug-and-play on top of two-stream networks. Therefore, we selected three baseline models. For each baseline, we provide detailed instructions on how to apply our method in the corresponding ReadMe files located in three different folders. Please refer to the respective ReadMe files for replication instructions.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{WTAL_Li_2024,
  title     = {Efficient Dual-Confounding Eliminating for Weakly-supervised Temporal Action Localization},
  author    = {Li, Ao and Liu, Huijun and Sheng, Jinrong and Chen, Zhongming and Ge, Yongxin},
  year      = 2024,
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  location  = {Melbourne VIC, Australia},
  address   = {New York, NY, USA},
  series    = {MM '24},
  pages     = {8179–8188},
  doi       = {10.1145/3664647.3681571},
  isbn      = 9798400706868,
  numpages  = 10,
  keywords  = {consistency mining, structural causal model, substituted confounder set, temporal action localization, weakly-supervised}
}
```

## Acknowledgement

We would like to express our sincere gratitude to the following authors for their contributions to the community:

 - [CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net)
 - [DELU](https://github.com/MengyuanChen21/ECCV2022-DELU)
 - [DDG-Net](https://github.com/XiaojunTang22/ICCV2023-DDGNet)