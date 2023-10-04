# Contributing

To contribute to the project, you might need extra requirements:

- [GNU Make](https://www.gnu.org/software/make/) for helper scripts.
- [Python 3.9](https://www.python.org/downloads/) for backend and ML.
- [Poetry](https://python-poetry.org/) for Python package and environment management.
- [Node 16](https://nodejs.org/en/download) for web application.

Once all requirements are installed, have a look in the commands provided by `make`.
It is recommended to start the core services (ray cluster, backend, mlflow and databases)
on docker and run the webapplication locally.

We have a series of git hooks to keep up with the code standards defined for the project.
To have them working in your machine run:

```bash
make pre-commit-install
```


## Running the code in docker-compose

- Start the stack with Docker Compose (only needed to apply the changes of configuration files, code changes are captured by a volume defined in the [override file](./docker-compose.override.yml).

This command also starts the [monitoring services](./CONTRIBUTING.md#monitoring).

```bash
docker compose up -d
```

- Update dependencies on a running container

```bash
docker compose exec backend poetry install
```

- Forces rebuilding of docker services:

```bash
docker compose up --build
```

- To check the logs and attach to it

```bash
docker compose logs --follow
```

- To check the logs of the backend, or some specific server

```bash
docker compose logs --follow backend
```

- Tune docker compose to your needs by changing the [override file](./docker-compose.override.yml) or ignore it with:

```bash
docker compose -f ./docker-compose.yml <...> # whatever docker compose commad you'd like to run without being overwritten by override file
```

## Automated Tests

Refer to [pytest documentation](https://docs.pytest.org/en/7.2.x/) to know how to best select what tests to run
If your stack is already up and you just want to run the tests, you can use:

- Run default tests (short run)

```bash
docker compose exec backend pytest
```

- Run tests with HTML coverage report

```bash
docker compose exec backend pytest --cov-report=html
```

- Run all tests (will take some time)

```bash
docker compose exec backend pytest -m 'not benchmark'
```

Some tests will fail if ran outside docker-compose.


### Backend

```bash
poetry shell # starts shell with the virtualenv loaded or ...
poetry run code . # starts vscode with the virtualenv loaded in this folder
```
#### Running Benchmarks

When implementing a benchmark, place it along other tests, and name your test function as `test_benchmark_name-of-benchmarking-target`, and mark it with `@pytest.mark.benchmark`

To execute a set of benchmarks, use the following pytest options:

- `-s` pytest option to capture the test stdout
- `-m benchmark` otherwise the benchmarks collected will be deselected by the pytest.ini `adopts`

## Live development with Python Jupyter Notebooks

To run jupyterlab withing the docker compose networks (in order to access services such as the db and ray), run:

```bash
docker compose exec backend bash # get a shell in container
$JUPYTER # use JUPYTER variable defined in docker-compose.override.yml
```

If calling $JUPYTER fails, possible causes are:

1. Docker-compose.override.yml was not used, in such case you can run in the backend container shell:

```bash
jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
```

2. Development dependencies were not installed, in such case you should
   get a shell in the container and run `poetry add jupyter`. The default
   development image (Dockerfile.local) already has jupyter as a dev dependency

## Migration

Here we talk about the migrations.

#### Running existing migrations

There are 3 make commands that many help with migrations, their names are self-explanatory:

1. `make migrate-backend`: Runs backend migrations
1. `make migrate-mlflow`: Runs mlflow migrations
1. `make migrate`: Runs all migrations

As during local development your app directory is mounted as a volume inside the container, you can also run the migrations with `alembic` commands inside the container and the migration code will be in your app directory (instead of being only inside the container). So you can add it to your git repository.

Make sure you create a "revision" of your models and that you "upgrade" your database with that revision every time you change them. As this is what will update the tables in your database. Otherwise, your application will have errors.

- Start an interactive session in the backend container:

```console
$ docker compose exec backend bash
```

- If you created a new model in `./mariner/entities`, make sure to import it in `mariner/entities/__init__.py`, that Python module that imports all the models will be used by Alembic.

- After changing a model (for example, adding a column), inside the container, create a revision, e.g.:

```console
$ alembic revision --autogenerate -m "Add column last_name to User model"
```

- Commit to the git repository the files generated in the alembic directory.

- After creating the revision, run the migration in the database (this is what will actually change the database):

```console
$ alembic upgrade head
```


## Publishing Release Notifications

A cli tool was made to publish release notifications from the [RELEASES.md](./RELEASES.md) file
into the database. (PS: This tool needs to be migrated to the `backend/cli/` folder.)

If on the commit, there is an **upgrade to the version in the pyproject**, the script will
detect it, and use it to filter the current release to publish notifications for.

You may use it from the command line for debugging like so:

```bash
cd backend
cat RELEASES.md | python -m mariner.changelog publish
```

or with make

```bash
make publish
```

## Testing the application

When you just want to test the application, i.e. you don't want to make changes to it, it's
recommended to run the project without src code volumes created by default with our docker-compose
files. To do that, you must run all make commands like following. :

```console
make build DOCKER_COMPOSE="docker-compose.yml"
make start-backend DOCKER_COMPOSE="docker-compose.yml"
```

Causing only the docker-compose.yml file to have affect, and therefore ignoring the volumes created in docker-compose.override.yml

## Monitoring and Observability

We have a `grafana` and `prometheus` services defined in `docker-compose.yml` that must be up to monitor the application.
Specifically, we monitor ray using their exported dashboards through a volume on ray-head `/tmp/ray/metrics/`
as documented [here](https://docs.ray.io/en/latest/ray-observability/ray-metrics.html).

The grafana credentials is `admin` and `123456` when running it thorugh docker compose.

Other metrics are yet to be exposed to prometheus so they can be visualized in grafana.

## Troubleshooting

- Getting pre-commit failures because of style when trying to commit

  Try running the following command to run automatic formatters:

  ```console
  make fix
  ```

- Got a: `OSError: libcublas.so.11: cannot open shared object file: No such file or directory`

  **CPU** users:

  Reinstall torch with cpu extra deps in the correct virtual environment

  ```
  poetry shell
  pip uninstall torch
  pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu
  ```
