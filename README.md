# MULE - Your music understanding workhorse

This codebase covers the MULE (**M**usicset **U**nsupervised **L**arge **E**mbedding) model and supporting materials for the ISMIR 2022 publication:

[*Supervised and Unsupervised Learning of Audio Representations for Music Understanding*](https://arxiv.org/abs/2210.03799), **M. C. McCallum**, F. Korzeniowski, S. Oramas, F. Gouyon, A. F. Ehmann.

It is intended for research purposes to better understand the results in the paper, and enable further research into audio embeddings, or downstream models that rely on audio or multimodal understanding of music content.

An overview of the training process for the MULE model is depicted below.

![MULE model](/img/MULEDiagram.png?raw=true)

# License

The mule python module is licensed under the [GNU GPL 3.0 license](https://www.gnu.org/licenses/gpl-3.0.en.html).

The supporting data, including:

 - MULE model weights
 - Additional results
 - Model and probe hyperparameter configurations

are licensed under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

# Installation and Use

MULE is hosted on pypi and may be installed with pip:

```
pip install sxmp-mule
```

This module requires FFMpeg to read audio files, which may be downloaded [here](https://ffmpeg.org/download.html).

MULE uses [SCOOCH](https://github.com/PandoraMedia/scooch) as a configuration interface, which relies on yaml configuration files. To see an example configuration for running mule, please check out the provided configuration [here](/supporting_data/configs/mule_embedding.yml).

**NOTE:** MULE model weights are stored on `git lfs`. If you wish to run a model using these weights, we recommend you clone this git repository:

```
git clone https://github.com/PandoraMedia/music-audio-representations.git
```

This repository contains both an example SCOOCH configuration and the MULE model weights, which may be used to analyze an audio file.

For example, to run MULE you can use the `mule` CLI in conjuction with a SCOOCH yaml file, which by default references the mule model weights relative to the root directory in this repository e.g.,

```
git clone https://github.com/PandoraMedia/music-audio-representations.git
cd ./music-audio-representations
mule analyze --config ./supporting_data/configs/mule_embedding.yml -i ./test.wav -o ./embedding.npy
```

# Supporting material

In addition to the python mule module, this repository provides **more detailed results** and **hyperparameter configurations** for the probes trained in the aforementioned publication.

**Results** tables can be found in `./supporting_data/results`.

SCOOCH **hyperparameter** configurations for the experiments in the publication can be found in `./supporting_data/configs/probes`.
