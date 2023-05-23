.. _overview
Overview
========

Mariner is ...

Architecture
------------
The main components of Mariner are the following:
- Backend: The backend is a REST API that exposes the functionality of Mariner. It is written in Python using the FastAPI framework.
  - mariner: The mariner package contains the core functionality of Mariner. It is written in Python.
  - fleet: The fleet package is an ongoing attempt to isolate the data science functions into a single module.
- Frontend: The frontend is a web application that allows users to interact with the backend. It is written in Javascript using the React framework.

Machine Supported Algorithms
----------------------------

Regression
~~~~~~~~~~
* :ref:`Neural Networks`.
* :ref:`Random Forest`.

Binary Classification
~~~~~~~~~~~~~~~~~~~~~
* :ref:`Neural Networks`.
* :ref:`Random Forest`.

Multiclass Classification
~~~~~~~~~~~~~~~~~~~~~~~~~
* :ref:`Neural Networks`.
* :ref:`Random Forest`.

Using the ray cluster
---------------------
Ray is a python framework for scaling ML tasks. We use throughout Mariner to scale the training of models.
Here is a non-extensive list of useful documentation to read before implementing or
reading into the way we use it.

Interacting with mlfow
----------------------
Mlflow is a tool to track experiments and models. We use it to track the experiments,
and manage model versions.


