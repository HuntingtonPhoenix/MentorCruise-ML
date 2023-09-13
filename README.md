<br/>
<p align="center">
  
  <h3 align="center">Quantization Testing</h3>

  <p align="center">
    Testing the effects of Quantization on various model's speed and accuracy.
    <br/>
    <br/>
  </p>
</p>

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Authors](#authors)

## About The Project

This project compares the effects of different compilation/quantization options on several standard models. The goals being to measure difference in speed an accuracy. 

## Built With

Machine Learning models and datasets were tested using the following major toolkits (see requirements.txt for full list.)

- Pytorch
- NVidia TensorRT
- TorchVision



## Getting Started


### Prerequisites

Your linux box must have the following preinstalled: 

- NVidia Graphics Card
- Python version 3.10.12 or later
- NVidia's TensorRT SDK from: https://developer.nvidia.com/tensorrt-getting-started


### Installation

1. Clone the repository
```sh
git clone https://github.com/HuntingtonPhoenix/MentorCruise-ML.git
```

2. Move to the cloned directory

3. Create a virtual environment
```sh
python3 -m venv ./.venv
```

4. Activate the virtual environment
```sh
 source .venv/bin/activate
```

5. Install the prerequisites using PIP
```sh
pip install -r requirements.txt 
```

6. Install the TensorRT libs using PIP
```sh
pip install --extra-index-url https://pypi.nvidia.com tensorrt-libs
```

6. Install the Torch TensorRT lib using PIP
```sh
pip3 install torch-tensorrt -f https://github.com/NVIDIA/Torch-TensorRT/releases
```


## Usage

Run:
```sh
python3 ClassModelSpeedTest.py
```

This will create an output directory is one doesn't already exist and write csv and png files for each model tested and their associated speed tests.

Accuracy is currently commented out for the gpu based tests because initial runs showed a less than .1% accuracy changes across all the options tested for each model. 

The CPU tests are currently unavailable as the 8Bit Quantization settings are incorrect leading to fast results, but accuracy near 0%. This needs to be investigated further. 

## License

Distributed under the MIT License. 

## Authors

* **Kevin Johnson** 
