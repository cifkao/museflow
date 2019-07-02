# museflow
museflow is an experimental music sequence learning toolkit, built on top of TensorFlow.

The most important modules are:
- `museflow.components` – building blocks for TensorFlow models (e.g. RNN decoder)
- `museflow.encodings` – classes defining ways to encode music for use with the models
- `museflow.trainer` – a basic implementation of model loading, saving and training
- `museflow.config` – a hierarchical configuration mechanism for setting up experiments
- `museflow.models` – implementations of basic models (accessible via the `museflow model` command)
- `museflow.scripts` – pre- and post-processing scripts (accessible via the `museflow script` command)

To install, run:
```sh
pip install '.[gpu]'
```
To install without GPU support:
```sh
pip install '.[nogpu]'
```

## Copyright notice
Copyright 2019 Ondřej Cífka of Télécom Paris, Institut Polytechnique de Paris.  
All rights reserved.
