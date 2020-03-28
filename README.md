:sunrise: Neural Style Transfer app
===================================
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/anomam/nst/blob/master/LICENSE)

This repository provides a Python implementation of Neural Style Transfer (by [Gatys et al. (2015)](https://arxiv.org/abs/1508.06576)) using Keras and the Tensorflow Adam optimizer.

The implementation comes with a streamlit application to provide a user-friendly interface that allows an easy selection of the training parameters as well as a convenient way to load input images, train, and explore the results.

Here is an example of running the app to transfer artistic features from [the Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night) to a picture of the Tokyo Tower.

<div align="center">
 <img src="https://github.com/anomam/nst/raw/master/outputs/example/NST_app_demo.gif" width="512px">
</div>

Some of the obtained results:

<div align="center">
 <img src="https://github.com/anomam/nst/raw/master/inputs/starry_night.jpg" height="200px">
 <img src="https://github.com/anomam/nst/raw/master/inputs/tokyo_tower.jpg" height="200px">
 <img src="https://github.com/anomam/nst/raw/master/outputs/example/starry_tokyo_tower.gif" width="512px">
</div>


## Contents
- [Installation requirements](#requirements)
- [Usage](#usage)
- [License](#license)


## Requirements
You have the option to run this implementation on either CPU or GPU, but in both cases using Python 3.6. It is recommended to use the GPU option for more speedy results.

### Running on CPU
This is the easiest approach to run the app.

- Install the python requirements using [pip](https://pypi.org/project/pip/), preferrably in a virtual environment: ``pip install -r requirements.txt``
- [CMake](https://cmake.org/install/): if you don't already have it, you'll need this to run the `Makefile` recipies

You can make sure that the installation is working by running the tests with

``` bash
make cpu-test-run
```

### Running on GPU
This will train a lot faster than using CPUs.

- [nvidia-docker2](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)): ``nvidia-docker2`` will allow to build a container with all the GPU requirements for training
- [CMake](https://cmake.org/install/): if you don't already have it, you'll need this to run the `Makefile` recipies

You can make sure that the installation is working by running the tests with

``` bash
make gpu-test-run
```

## Usage

In order to start the app, you just need to run a single command line from the root folder of this repository as follows:

- for running on CPU

``` bash
make cpu-app
```

- for running on GPU

``` bash
make gpu-app
```

## License

See [LICENSE](LICENSE) for details.
