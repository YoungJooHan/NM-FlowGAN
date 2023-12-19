# NM-FlowGAN
- Han, Young-Joo, and Ha-Jin Yu. "NM-FlowGAN: Modeling sRGB Noise with a Hybrid Approach based on Normalizing Flows and Generative Adversarial Networks." arXiv preprint arXiv:2312.10112 (2023). [Arxiv](https://arxiv.org/abs/2312.10112)

----
## Abstract
Modeling and synthesizing real sRGB noise is crucial for various low-level vision tasks. The distribution of real sRGB noise is highly complex and affected by a multitude of factors, making its accurate modeling extremely challenging. Therefore, recent studies have proposed methods that employ data-driven generative models, such as generative adversarial networks (GAN) and Normalizing Flows. These studies achieve more accurate modeling of sRGB noise compared to traditional noise modeling methods. However, there are performance limitations due to the inherent characteristics of each generative model. To address this issue, we propose NM-FlowGAN, a hybrid approach that exploits the strengths of both GAN and Normalizing Flows. We simultaneously employ a pixel-wise noise modeling network based on Normalizing Flows, and spatial correlation modeling networks based on GAN. In our experiments, our NM-FlowGAN outperforms other baselines on the sRGB noise synthesis task. Moreover, the denoising neural network, trained with synthesized image pairs from our model, also shows superior performance compared to other baselines.

----
## Requirements
- Python 3.11.3
- h5py 3.7.0
- PyTorch 1.13.0
- NumPy 1.22.3
- Einops 0.6.0

----
## Data Preparation
Follow below descriptions to build code directory.
```
NM-FlowGAN
├─ ckpt
├─ conf
├─ data
│  ├─ SIDD
│  │  ├─ SIDD_Medium_Srgb
│  │  ├─ BenchmarkNoisyBlocksSrgb.mat
│  │  ├─ ValidationGtBlocksSrgb.mat
│  │  ├─ ValidationNoisyBlocksSrgb.mat
|  │  ├─ HDF5_s96_o48
├─ output
├─ core
├─ util
```
- Make `data` directory and place the [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) dataset.
- Make `HDF5_s96_o48` directory by using `prep.sh`.

----
## Test Pretrained Model
Please download the noise synthesizing model and denoising model trained on SIDD medium from this [Link:NoiseModel](https://drive.google.com/file/d/1WaJg2ykB5k1KTd53zm7S9_WPSvg37Sfy/view?usp=sharing), [Link:DenoisingModel](https://drive.google.com/file/d/1W_34cuQvGRAjyCMwbLj3JZX2aoRbvboE/view?usp=sharing). Put it in path:

```
/ckpt/NMFlowGAN_NoiseModeler.pth
/ckpt/NMFlowGAN_Denoiser.pth
```

Example: Test noise synthesizing with our pretrained model for the SIDD dataset.
```
python test.py --config=./conf/NMFlowGAN_NoiseModeler.yaml BASE.pretrained NMFlowGAN_NoiseModeler.pth
```

Example: Test image denoising with our pretrained model for the SIDD dataset.
```
python test.py --config=./conf/NMFlowGAN_Denoiser.yaml BASE.pretrained NMFlowGAN_Denoiser.pth
```

----
## Training & Test Your Model
### Noise Synthesizing
Example: Train our noise model for the SIDD dataset.
```
python train.py --config=./conf/NMFlowGAN_NoiseModeler.yaml LOG.session_name [SESSION_NAME]
```

Example: Test our noise model for the SIDD dataset.
```
python test.py --config=./conf/NMFlowGAN_NoiseModeler.yaml LOG.session_name [SESSION_NAME] TEST.ckpt_epoch [EPOCHS] 
```


### Denoising
Example: Train our denoiser by using pretrained noise model for the SIDD dataset.
```
python train.py --config=./conf/NMFlowGAN_Denoiser.yaml MODEL.DNCNNFLOWGAN.pretrained_path [*.pth file path of noise modeling network]
```

Example: Test our denoiser for the SIDD dataset.
```
python test.py --config=./conf/NMFlowGAN_Denoiser.yaml LOG.session_name [SESSION_NAME] TEST.ckpt_epoch [EPOCHS] 
```

---
## Acknowledgement
The codes are based on [AP-BSN](https://github.com/wooseoklee4/AP-BSN),  [Noise2NoiseFlow](https://github.com/SamsungLabs/Noise2NoiseFlow). Thanks for the great works.
