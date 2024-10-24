# FDF: Flexible Decoupled Framework for Time Series Forecasting with Conditional Denoising and Polynomial Modeling

[![GitHub Stars](https://img.shields.io/github/stars/zjt-gpu/FDF.svg)](https://github.com/zjt-gpu/FDF/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/zjt-gpu/FDF.svg)](https://github.com/zjt-gpu/FDF/network/members)

> **Abstract:** Time series forecasting is vital in numerous web applications, influencing critical decision-making across industries. While diffusion models have recently gained increasing popularity for this task, we argue they suffer from a significant drawback: indiscriminate noise addition to the original time series followed by denoising, which can obscure underlying dynamic evolving trend and complicate forecasting. To address this limitation, we propose a novel flexible decoupled framework (FDF) that learns high-quality time series representations for enhanced forecasting performance. A key characteristic of our approach leverages the inherent inductive bias of time series data of its decomposed trend and seasonal components, each modeled separately to enable decoupled analysis and modeling. Specifically, we propose an innovative Conditional Denoising Seasonal Module (CDSM) within the diffusion model, which leverages statistical information from the historical window to conditionally model the complex seasonal component. Notably, we incorporate a Polynomial Trend Module (PTM) to effectively capture the smooth trend component, thereby enhancing the model's ability to represent temporal dependencies. Extensive experiments validate the effectiveness of our framework, demonstrating superior performance over existing methods and highlighting its flexibility in time series forecasting.
![](picture/model.png)

# Main Results
![](picture/results.png)

## Requirements
torch==2.4.1
pandas==2.2.3
scikit-learn==1.5.2
timm==1.0.10
einops==0.8.0
reformer_pytorch==1.4.4
wandb==0.18.3
pytorch_lightning==2.4.0
opt_einsum==3.4.0
linear_attention_transformer==0.19.1
matplotlib==3.9.2
thop==0.1.1.post2209072238

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

We use the following five real-world datasets for our experiments. They are placed in the `./datasets` folder in the repository. Please ensure you adhere to each dataset's respective license when using them.

1. **ETTh**: *Informer: Beyond efficient transformer for long sequence time-series forecasting*. Available at [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17325).
2. **Exchange**: *Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*. Available at [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3209978.3210006).
3. **Weather**: Weather data is available at [https://www.bgc-jena.mpg.de/wetter/](https://www.bgc-jena.mpg.de/wetter/).
4. **Electricity**: Electricity consumption data is available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014/).
5. **Wind**: *Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement*. Available at [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/91a85f3fb8f570e6be52b333b5ab017a-Abstract-Conference.html).



## Usage

Commands for training and testing the FDF on Dataset ETTh1 and ETTh2 respectively:
```bash
# ETTh1
python -u run.py --model_name FDF --dataset ETTh1
More parameter information please refer to `args.py`.
# ETTh2
python -u run.py --model_name FDF --dataset ETTh2
More parameter information please refer to `args.py`.
```

## Acknowledgement

We extend our sincere appreciation to the following GitHub repositories for providing invaluable codebases:

[DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)
[CSDI](https://github.com/ermongroup/CSDI)
[Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS)
[D3VAE](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE)
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)

## Citation
If you find this repository useful for your work, please consider citing it as follows:

```bibtex
@misc{zhang2024fdf,
  title={FDF: Flexible Decoupled Framework for Time Series Forecasting with Conditional Denoising and Polynomial Modeling},
  author={Zhang, Jintao and Cheng, Mingyue and Tao, Xiaoyu and Liu, Zhiding and Wang, Daoyu},
  year={2024},
  eprint={2410.13253},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2410.13253}, 
}
```