# Getting started

## Installation

To run the application, you simply need:

- [Docker](https://www.docker.com/).
- [Docker Compose](https://docs.docker.com/compose/install/).
- and optionally [GNU Make](https://www.gnu.org/software/make/), which is included in many distributions, and will allow you to use shorter commands


## Starting the application locally:

Use one of the following:

- `make start` is the fastest way to start all the needed applications.
- `docker compose up --wait backend webapp` is also another way that does not require building.

Create a local user to interact with the app (if OAUTH environment variables were not passed to you):
- `make create-admin`. This will create a user with email `admin@.mariner.trident.bio` and password `123456`.

Finally, access <http://localhost:3000>, login to use the app.


## Making code contributions:

See [the Contributing page](./CONTRIBUTING.md).