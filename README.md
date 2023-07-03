# Climbing Holds Classification

## Agenda

- [Background](#background)
- [Dataset](#dataset)
- [Prerequisites](#section-2)
- [Files in the repository](#section-3)
- [Running Instructions](#section-4)

## Background  <a name="background"></a>
our project is to develop a segmentation and classification system for climbing holds in a climbing wall. We aim to accurately classify climbing holds into their respective labels, including crimp, pocket, jug, pinch, and sloper. Additionally, we aim to provide precise instance segmentation masks for each climbing hold, enabling a detailed understanding of their spatial distribution on the climbing wall. By achieving high accuracy in both classification and segmentation, we aim to enhance the analysis of climbing holds and provide valuable insights for climbers, route setters, and training programs. 

![types](https://github.com/orilevi2809/DL/assets/62295757/15da2956-b4ad-41cd-a87e-c91f93f70dec)

<a name="dataset"></a>
## Dataset  
After collecting images for our dataset, we performed accurate labeling, resulting in the following distribution: 183 images were labeled as 'crimp', 207 images as 'pinch', 234 images as 'pocket', 247 images as 'sloper', and 209 images as 'jug'. These labeled images provide a diverse and balanced representation of different climbing holds, enabling us to train our model effectively on a range of grip types.



<a name="section-2"></a>
## Prerequisites

| Library       | Version |
|---------------|---------|
| sklearn       | 0.24.2  |
| torchvision  | 0.10.0  |
| torch         | 1.9.0   |
| PIL           | 8.3.1   |
| matplotlib    | 3.4.3   |
| numpy         | 1.21.0  |
| cv2           | 4.5.3   |
| detectron2    | 0.5     |

<br/><br/>
To install Detectron2, you should run the following commands:
```shell
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
<br/><br/>
We recommend installing CUDA for optimal performance. You can refer to the [System Requirements](https://docs.nvidia.com/cuda/archive/11.8.0/pdf/CUDA_Installation_Guide_Windows.pdf) provided by NVIDIA for CUDA installation.

For the installation guide of CUDA, you can visit the following link:
[Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

To install Torch and torchvision, please visit the official [PyTorch website](https://pytorch.org/) for the appropriate installation command.

<a name="section-3"></a>
## Files in the repository
| File name     | Purpose                                          |
|---------------|--------------------------------------------------|
| mobileNet.py  | Load the data, train our network, and save our trained model |
| main.py       | Get the image and classify the climbing holds with the saved model |


<a name="section-4"></a>
## Running Instructions

<a name="section-5"></a>
## Results

![WhatsApp Image 2023-06-28 at 17 44 37](https://github.com/orilevi2809/DL/assets/62295757/921eb0b0-1b45-4db1-a7f2-352412cb416d)         ![WhatsApp Image 2023-06-28 at 17 45 23](https://github.com/orilevi2809/DL/assets/62295757/77801e4d-9a56-43aa-9fbf-443e680fca53)










