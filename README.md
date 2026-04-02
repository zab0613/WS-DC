# WS-DC
This repository is the official implementation of the paper:
* Spectral Efficiency-Aware Codebook Design for Task-Oriented Semantic Communications
* Authors: Anbang Zhang, Shuaishuai Guo, Chenyuan Feng, Shuai Liu, Hongyang Du, Geyong Min
  
## Design Philosophy
### This Work: Spectral Efficiency-Aware Codebook Design for Task-Oriented Semantic Communications
<img width="1460" height="629" alt="image" src="https://github.com/user-attachments/assets/21931745-6509-4819-a798-0651ca63de9d" />
Digital task-oriented semantic communication (ToSC) aims to transmit only task-relevant information, significantly reducing communication overhead. Existing ToSC methods typically rely on learned codebooks to encode semantic features and map them to constellation symbols. However, these codebooks are often sparsely activated, resulting in low spectral efficiency and underutilization of channel capacity. This highlights a key challenge: how to design a codebook that not only supports task-specific inference but also approaches the theoretical limits of channel capacity. To address this challenge, we construct a spectral efficiency-aware codebook design framework that explicitly incorporates the codebook activation probability into the optimization process. Beyond maximizing task performance, we introduce the Wasserstein (WS) distance as a regularization metric to minimize the gap between the learned activation distribution and the optimal channel input distribution. Furthermore, we reinterpret WS theory from a generative perspective to align with the semantic nature of ToSC. Combining the above two aspects, we propose a WS-based adaptive hybrid distribution scheme, termed WS-DC, which learns compact, task-driven and channel-aware latent representations. Experimental results demonstrate that WS-DC not only outperforms existing approaches in inference accuracy but also significantly improves codebook efficiency, offering a promising direction toward capacity-approaching semantic communication systems.

## Training
Firstly, train the UIS-ToSC framework.
```
python /run_VQVAE.py
----main_train()
```
Then, train the attacker stealing models.
```
python /run_VQVAE.py
----main_test()
```

## Simulation dataest
CIFAR-10 dataest/Tiny-Imagenet

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
