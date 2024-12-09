[ACM MM24 (Oral)] DMFourLLIE: Dual-Stage and Multi-Branch Fourier Network for Low-Light Image Enhancement [Paper](https://arxiv.org/abs/2412.00683)

## Abstract
We propose a Dual-Stage Multi-Branch Fourier Low-Level Image Enhancement (DMFourLLIE) framework. In the first stage, DMFourLLIE integrates infrared image structure, spatial details, and complete Fourier information to enhance the overall image brightness, while avoiding potential issues introduced by independent component restoration. In the second stage, due to the limitations of mainly preserving spatial information in the phase component, we transform the inherent phase component stage into a pixel reconstruction stage. Specifically, it involves a dual-branch architecture with multi-scale spatial perception and Fast Fourier Convolution, enabling robust recovery of spatial structure and subtle texture variations in a single holistic optimization.

## Over-all-Architecture
![Over-all-Architecture](https://github.com/user-attachments/assets/e73e3d11-9b35-4363-a066-d399701414f2)

## Installation
```
conda env create -f environment.yml
conda activate DMFourLLIE
```

## Train
```python train.py -opt ./options/train/huawei.yml```

## Test
```python test.py -opt ./options/test/huawei.yml```

## Datasets
- LOL-real and LOL-sys can be found in [here](https://github.com/flyywh/SGM-Low-Light).
- LSRW-Huawei can be found in [here](https://github.com/JianghaiSCU/R2RNet).

## Infrared Image Generation
- The generation of infrared images refers to [here](https://github.com/RPM-Robotics-Lab/sRGB-TIR).

## Acknowledgement
This repo is based on [FourLLIE](https://github.com/wangchx67/FourLLIE).

## Citation Information
If you find the project useful, please cite:  

```bibtex  
@inproceedings{zhang2024dmfourllie,  
    title={DMFourLLIE: Dual-Stage and Multi-Branch Fourier Network for Low-Light Image Enhancement},  
    author={Zhang, Tongshun and Liu, Pingping and Zhao, Ming and Lv, Haotian},  
    booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},  
    pages={7434--7443},  
    year={2024}  
}  
