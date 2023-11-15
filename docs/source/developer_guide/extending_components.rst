
=====================
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

