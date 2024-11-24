# DMFourLLIE_ACMMM24_ORAL
DMFourLLIE: Dual-Stage and Multi-Branch Fourier Network for Low-Light Image Enhancement, ACM MM24 (Oral)

# Abstract
In the Fourier space, some researchers have noticed that brightness information is primarily reflected in the amplitude component, while spatial information mainly exists in the phase component. Therefore, existing methods commonly adopt a two-stage restoration framework that separately recovers the amplitude and phase components. We further explore how to effectively utilize Fourier space information and find that decoupling the inherent correlations in Fourier space is not the optimal choice, and the color distortion and noise issues introduced in the first stage by previous methods cannot be effectively addressed in the second stage through phase space information. To this end, we propose a Dual-Stage Multi-Branch Fourier Low-Level Image Enhancement (DMFourLLIE) framework. In the first stage, DMFourLLIE integrates infrared image structure, spatial details, and complete Fourier information to enhance the overall image brightness, while avoiding potential issues introduced by independent component restoration. In the second stage, due to the limitations of mainly preserving spatial information in the phase component, we transform the inherent phase component stage into a pixel reconstruction stage. Specifically, it involves a dual-branch architecture with multi-scale spatial perception and Fast Fourier Convolution, enabling robust recovery of spatial structure and subtle texture variations in a single holistic optimization. Extensive experiments demonstrate that DMFourLLIE outperforms state-of-the-art techniques on three different datasets. Importantly, DMFourLLIE achieves enhanced restoration capabilities while maintaining parameter efficiency. Our code will be available after acceptance.

# Over-all-Architecture
![Over-all-Architecture](https://github.com/user-attachments/assets/e73e3d11-9b35-4363-a066-d399701414f2)

# Phase1
![phase1](https://github.com/user-attachments/assets/646a6ff9-3d4e-4f02-a9c8-72fdd934fa36)

# phase2
![phase2](https://github.com/user-attachments/assets/0da91527-ec98-4eb9-980b-889723e89fff)

# Train
'''python train.py -opt ./options/train/LOLv2_real.yml

# Test
'''python test.py -opt ./options/test/LOLv2_real.yml

# Acknowledgement
This repo is based on [FourLLIE](https://github.com/wangchx67/FourLLIE).

# Citation Information
If you find the project useful, please cite:
@inproceedings{zhang2024dmfourllie,
  title={DMFourLLIE: Dual-Stage and Multi-Branch Fourier Network for Low-Light Image Enhancement},
  author={Zhang, Tongshun and Liu, Pingping and Zhao, Ming and Lv, Haotian},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={7434--7443},
  year={2024}
}
