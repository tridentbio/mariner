.. _modelschema:

===============
Creating Models
===============

Machine Learning models built in Mariner are composed of smaller pieces that we'll call components. There are 4 kinds of components:

- transforms and featurizers. They are used to preprocess the data before it is fed to the model.

- torch modules (`nn.Module`). They are used to create neural networks.

- sklearn models. Used to create non-neural models provided by sklearn, which provides a good starting point for most tasks.

Here we explain how to add a new featurizer, transformer, torch layer or sklearn model. Those components can be serialized and used by model specifications explained below.


YAML Model Schemas
==================

All model specs can be represented in YAML or JSON. For readability, we use YAML in most of the examples of working models in the `tests folder <https://github.com/tridentbio/mariner/tree/develop/backend/tests/data/yaml>`_, and JSON when using the REST API.

In this section we'll clarify some important aspects of the specs.

``forward_args`` property
-------------------------

The ``forward_args`` is used by components to specify it's inputs and build a directed acyclic graph that allows to specify the computational model that will be used with the model and preprocessing pipeline.
We refer to the outputs of previous components by reference with the ``$`` character.

``framework`` property
----------------------

Specifies what framework should be used with the model. Currently, we support ``torch`` and ``sklearn``. Depending on the value of this property, a different ``spec`` property is used to specify the model.

``dataset`` property
--------------------

The ``dataset`` is used to specify the dataset. It's ``name`` property should refer to an existing dataset, i.e. one created previously through the API. Also, the ``data_type`` property specified in ``feature_columns`` and ``target_columns`` should be one of the data types provided by :doc:`/generated/fleet.data_types`.

Inside of ``dataset`` we also specify the preprocessing pipeline. The ``dataset.strategy`` property controls the schema used to specify the pipeline and can be one of the following:

- ``pipeline``: Where the featurizer and transform of each column is specified along with it. This is a more compact format that was included with sklearn support. An example can be seen in `Example of a sklearn model`_.
- ``forward_args``: The preprocessing pipeline is specified like the ``forward_args`` property of the model. This is the default strategy and the one we recommend using. An example can be seen in `Example of a torch model`_.


``spec`` property
-----------------

This property describes the model, and may reference other components by name. The ``spec`` property is different for each framework. See the examples below to see how it is used.

Example of a torch model
------------------------

.. literalinclude:: ../../../backend/tests/data/yaml/small_regressor_schema.yaml

Example of a sklearn model
--------------------------

.. literalinclude:: ../../../backend/tests/data/yaml/sklearn_hiv_random_forest_classifier.yaml

