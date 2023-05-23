Mariner
===================================


Using Makefile
--------------

With `make` you can build, test and run the app.

Some utility commands are also available.

Run `make help` to see all available commands.


Configuring .env
----------------

Environment variables are divided into 2 files, `backend/.env` and `backend/.env.secret`.
The separation was made to support some CI workflows, but all variables should be considered
sensitive in production.

The `.env` file contains all variables that are not sensitive, and can be shared with the team.

The `.env.secret` file contains all sensitive variables, and should be kept secret.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   :doc:`Overview <overview>`
   :doc:`Model Schemas <modelschema>`


Indices and tables
==================
* :doc:`Overview <overview>`
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

