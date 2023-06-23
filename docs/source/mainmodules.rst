.. _api_ref:

=============
API Reference
=============

.. note::

   This is a work in progress.


This is a distilled version of the API reference.

.. _users_ref:

:mod:`mariner.users`: Users
===========================

.. automodule:: mariner.users
    :no-members:
    :no-inherited-members:

Classes
------------
.. currentmodule:: mariner

.. autosummary::
   :nosignatures:
   :toctree:
   :template: class.rst

   users.GithubAuth

Functions
---------
.. currentmodule:: mariner

.. autosummary::
   :toctree:
   :template: function.rst

   users.authenticate
   users.get_user
   users.update_user
   users.create_user_basic
   users.get_users

:mod:`mariner.datasets`: Datasets
=================================

.. automodule:: mariner.datasets
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: mariner

.. autosummary::
   :toctree:
   :template: function.rst

   datasets.get_my_datasets
   datasets.get_my_dataset_by_id
   datasets.create_dataset
   datasets.update_dataset
   datasets.delete_dataset
   datasets.parse_csv_headers
   datasets.get_csv_file
   datasets.process_dataset
   datasets.start_process


:mod:`mariner.models`: Models
=============================

.. automodule:: mariner.models
    :no-members:
    :no-inherited-members:

Classes
------------
.. currentmodule:: mariner

.. autosummary::
   :nosignatures:
   :toctree:
   :template: class.rst


Functions
---------
.. currentmodule:: mariner

.. autosummary::
   :toctree:
   :template: function.rst

   models.check_model_step_exception
   models.create_model
   models.get_models
   models.get_model_options
   models.get_model_prediction
   models.get_model
   models.delete_model


:mod:`mariner.experiments`: Experiments
=======================================

.. automodule:: mariner.experiments
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: mariner

.. autosummary::
   :nosignatures:
   :toctree:
   :template: class.rst


Functions
---------
.. currentmodule:: mariner

.. autosummary::
   :toctree:
   :template: function.rst

   experiments.get_metrics_for_monitoring
   experiments.get_optimizer_options
   experiments.get_experiments_metrics_for_model_version
   experiments.send_ws_epoch_update
   experiments.get_running_histories
   experiments.log_hyperparams
   experiments.log_metrics
   experiments.get_experiments
   experiments.create_model_training
   experiments.handle_training_complete



:mod:`mariner.deployment`: Deloyments
=====================================

.. automodule:: mariner.deployment
    :no-members:
    :no-inherited-members:

