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


Adding new components
=====================

Here we describe how to add new components to the Mariner framework. The more components we get supported, more models we can build.

Adding new featurizer or transformer
------------------------------------

First you'll need a pydantic model that serializes and validates the component. They are placed in the ``fleet`` package (see :doc:`fleet.preprocessing </generated/fleet.preprocessing>` Config classes for examples). Below we have an example of a pydantic model that serializes the ``FPVecFilteredTransformer`` from the ``molfeat`` package.

.. code-block:: python

  class FPVecFilteredTransformerConstructorArgs(BaseModel):
      """
      Models the constructor arguments of a FPVecFilteredTransformer.
      """

      del_invariant: bool = False
      length: int = 2000


  @options_manager.config_featurizer()
  class FPVecFilteredTransformerConfig(CreateFromType, TransformConfigBase):
      """
      Models the usage of FPVecFilteredTransformer.
      """

      name: str
      constructor_args: FPVecFilteredTransformerConstructorArgs = (
          FPVecFilteredTransformerConstructorArgs()
      )
      type: Literal[
          "molfeat.trans.fp.FPVecFilteredTransformer"
      ] = "molfeat.trans.fp.FPVecFilteredTransformer"
      forward_args: dict


``options_manager`` is a singleton exported from :doc:`/generated/fleet.options` that provides decorators and work by collecting the components pydantic classes, and making them available in the ``ComponentOptionsManager`` class.
All such classes must have a string ``name`` and string literal ``type`` with the full name of the python class that performs the transform/featurization.

Adding new torch layer
----------------------

Torch layers can be added the same as transforms and featurizers, but there are utilities for generating code.
There are some caveats when using the code generation scripts: code generation relies on correct type-hint annotations,
and ignores all non-primitive arguments when creating the ``forward_args`` and ``constructor_args`` pydantic models.
The module responsible for the code generation is the :doc:`/generated/fleet.model_builder.generate`. To add new layers for code generation,
we must add a ``Layer`` instance on the ``layers`` or ``featurizers`` lists.

Adding new sklearn class
------------------------

Currently, sklearn models must be added manually. They are placed in the :doc:`/generated/fleet.scikit_.schemas` module.
Differently from the torch models, the sklearn components don't have to specify forward args as they are inferred from the preprocessing pipeline graph
represented in the ``dataset`` property as explained previously.
The feature outputs of the pipeline are concatenated and fed as the ``X`` argument of the ``fit`` method of the sklearn model and target outputs are fed as the ``y`` argument of the ``fit`` method.

