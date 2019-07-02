# museflow
museflow is an experimental music sequence learning toolkit, built using TensorFlow.

The most important subpackages and modules are:
- `museflow.components` – building blocks for TensorFlow models (e.g. RNN decoder)
- `museflow.encodings` – classes defining ways to encode music for use with the models
- `museflow.trainer` – a basic implementation of model loading, saving and training
- `museflow.config` – a hierarchical configuration mechanism for setting up experiments
- `museflow.models` – implementations of basic models (accessible via the `museflow model` command)
- `museflow.scripts` – pre- and post-processing scripts (accessible via the `museflow script` command)

Run `pip install .` to install.
