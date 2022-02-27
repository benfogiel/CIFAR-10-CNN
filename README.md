# CNN for the CIFAR-10 Dataset

- Simple Convolutional Neural Network optimized for the CIFAR-10 Dataset
- Currently achieving over 85 % accuracy after 50 epochs

Usage (Google Colab):
- Visit [this](https://colab.research.google.com/github/benfogiel/CIFAR-10-CNN/blob/main/cifar.ipynb) link to open the Google Colab
- Make sure you're using the GPU: Edit>Notebook settings>Hardware accelerator>GPU
- Have Fun!

Usage (Locally):
- Clone this repo
- Working from the main repo directory, run ```conda env create -f environment.yml```. This will create a virtual environment containing the necessary dependencies
- Activate the virtual environment by running ```conda activate cifar``` (Anaconda is required)
- Then train and validate the model by running ```python3 main.py```
