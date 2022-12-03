# MULE - Your music understanding workhorse

This codebase covers the MULE (**M**usicset **L**arge **U**nsupervised **E**mbedding) model and supporting materials for the ISMIR 2022 publication:

*Supervised and Unsupervised Learning of Audio Representations for Music Understanding*, M. C. McCallum, F. Korzeniowski, S. Oramas, F. Gouyon, A. F. Ehmann.

It is intended for research purposes to better understand the results in the paper, and enable further research on audio embeddings or downstream models that rely on audio or multimodal understanding of music content.

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

MULE uses SCOOCH as a configuration interface, which relies on yaml configuration files. To see an example configuration for running mule, please check out the provided configuration [here](/configs/mule_embedding.yml).

To run MULE, you can use the CLI in conjuction with a SCOOCH yaml file, e.g.,

```
mule analyze --config ./configs/mule_embedding.yml -i ./test.wav -o ./embedding.npy
```

# Supporting material

In addition to the python mule module, this repository provides **more detailed results** and **hyperparameter configurations** for the probes trained in the aforementioned publication.

**Reults** tables can be found in `./supporting_data/results`.

SCOOCH **hyperparameter** configurations can be found in `./supporting_data/configs/probes`.
