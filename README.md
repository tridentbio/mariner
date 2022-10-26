# mariner

## Backend local development

### Requirements

- [Docker](https://www.docker.com/).
- [Docker Compose](https://docs.docker.com/compose/install/).
- [Poetry](https://python-poetry.org/) for Python package and environment management.

### Installation

1. (Linux users) Create a virtual environment with python3.9

```bash
poetry env use $(which python3.9)
```

2. Install the dependencies in the repo's root folder

```bash
poetry install && poetry run install_deps_cpu
```

3. (optional) Install pre-commit

```bash
poetry run pre-commit install
```

4. Use the virtualenv as appropriate. As long as in the virtualenv, all dependencies should be found, and editor support should have autocompletion working:

```bash
poetry shell # starts shell with the virtualenv loaded
poetry run code . # starts vscode in this folder
```

### Running the code in docker-compose

- Start the stack with Docker Compose (only needed to apply the changes of configuration files, code changes are captured by a volume defined in the [override file](./docker-compose.override.yml)

```bash
docker-compose up -d
```

- Update dependencies on a running container

```bash
docker-compose exec backend poetry install
```

- Forces rebuilding of docker services:

```bash
docker-compose up --build
```

- To check the logs and attach to it

```bash
docker-compose logs --follow
```

- To check the logs of the backend, or some specific server

```bash
docker-compose logs --follow backend
```

- Tune docker-compose to your needs by changing the [override file](./docker-compose.override.yml) or ignore it with:

```bash
docker-compose -f ./docker-compose.yml <...> # whatever docker-compose commad you'd like to run without being overwritten by override file
```

### Automated Tests

Refer to [pytest documentation](https://docs.pytest.org/en/7.2.x/) to know how to best select what tests to run
If your stack is already up and you just want to run the tests, you can use:

- Run default tests (short run)
```bash
docker-compose exec backend pytest
```

- Run tests with HTML coverage report

```bash
docker-compose exec backend pytest --cov-report=html
```

- Run all tests (will take some time)

```bash
docker-compose exec backend pytest -m 'not benchmark'
```

Some tests will fail if ran outside docker-compose.

#### Running Benchmarks

When implementing a benchmark, place it along other tests, and name your test function as `test_benchmark_name-of-benchmarking-target`, and mark it with `@pytest.mark.benchmark`

To execute a set of benchmarks, use the following pytest options:

- `-s` pytest option to capture the test stdout
- `-m benchmark` otherwise the benchmarks collected will be deselected by the pytest.ini `adopts`


### Live development with Python Jupyter Notebooks

> TODO!

To run jupyterlab withing the docker-compose networks (in order to access services such as the db and ray), run:

```bash
docker-compose up -d jupyterlab
```

jupyterlab is a service defined in the docker-compose that uses the same environment as the backend, by installs and runs jupyterlab on top of it

### Migrations

As during local development your app directory is mounted as a volume inside the container, you can also run the migrations with `alembic` commands inside the container and the migration code will be in your app directory (instead of being only inside the container). So you can add it to your git repository.

Make sure you create a "revision" of your models and that you "upgrade" your database with that revision every time you change them. As this is what will update the tables in your database. Otherwise, your application will have errors.

- Start an interactive session in the backend container:

```console
$ docker-compose exec backend bash
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

## Publishing releases

A cli tool was made to publish releases from the [RLEASES.md](./RELEASES.md) file.

If on the commit, there is an upgrade to the version in the pyproject, the script will
detect it, and use it to filter the current release to publish notifications for.

You may use it from the command line for debugging like so:

```bash
python -m app.changelog --help
cat RELEASES.md | python -m app.changelog publish
```

or from the docker compose

```bash
docker-compose run backend python -m app.changelog --help
cat RELEASES.md | docker-compose run backend python -m app.changelog publish
```
