# Photo Blur Semantic Segmentation
Note: This model is based off of EMANet by Li et al.
The original repo can be found here:
https://github.com/XiaLiPKU/EMANet

## Installation
Steps are adapted from EMANet, but updated to be more descriptive and support the outcrop dataset.
1. (Recommended): Create a python virtual environment for this project by running `python -m venv env`
    - Make sure to activate the environment before proceeding to the next steps
2. Run `pip install -r requirements.txt` to install prerequisite packages
3. Install pytorch with CUDA support. This project is tested with version 1.8.2, but the latest release should work. Directions can be found [here](https://pytorch.org/get-started/locally/).
4. Download the pretrained [ResNet50](https://drive.google.com/file/d/1ibhxxzrc-DpoHbHv4tYrqRPC1Ui7RZ-0/view?usp=sharing) and [ResNet101](https://drive.google.com/file/d/1de2AyWSTHsZQRB_MI-VcOfeP8NAs3Wat/view?usp=sharing), unzip them, and put into the 'models' folder from the root directory of this project (you will have to create it).
5. **IMPORTANT:** Follow the steps in one of the sections below corresponding to the dataset you want to use.
6. Run `sh clean.sh` to clear the models and logs from the last experiment.
