# Contributing

To contribute to the project you'll need more requirements than those needed for just running the application:

- [GNU Make](https://www.gnu.org/software/make/) for common development tasks. 
- [Python 3.10](https://www.python.org/downloads/) to have full support of developer tools when testing code changes.
- [Poetry](https://python-poetry.org/) for Python package and environment management.
- [Node 16](https://nodejs.org/en/download) to have full support of developer tools when testing code changes.

Once all requirements are installed, have a look in the commands provided by `make`. You can do so by simply running `make` in the repository root. Please refer to the Makefile whenever we specify only a `make <...>` command for some task and you're curious to see what command is actually ran on the system.

We have a series of git hooks to keep up with the code standards defined for the project.
To have them working in your machine run:

```bash
make pre-commit-install
```

## Docker compose services

Running the application locally is easy with docker compose. If you're unfamiliar with it check out it's [user guide](https://docs.docker.com/compose/).

Now we describe the services available from our `docker-compose.yml` file.

We group the services into two different groups in the `Makefile`. This is helpful to spend less energy by not running services you won't use. First we have the core services, listed in `CORE_SERVICES` variable in `Makefile`, which are the services needed to have a complete interaction with the application, and it includes:

- `backend`: The REST API responsible to carry out the business rules of the application.
- `webapp`: The user front end that provides a graphical interface to use the application.
- `mlflow`: The MLflow service we have integrated with our functions.
- `mlflowdb` and `db`: Postgres databases used for the backend and mlflow respectively.
- `ray-head` and `ray-worker`: The ray head node and ray-worker nodes. They are used for simulating the production environment to provide the software to leverage the infrastructure to support scaling the application horizontally [^1].

We also have the monitoring services, which are services to provide monitoring and observability capabilities for developers:

- `grafana`: A tool to visualize metrics from the services.
- `prometheus`: A tool to collect metrics from the running services. The metrics are used by exposed by prometheus and used by grafana.


You'll notice other docker-compose files in the repository. Those are used to override or add docker-compose file properties depending on the environment where the app is running.

- `docker-compose-gpu.yml` contains the compose file overrides to run the backend with a computer that has access to GPU/CUDA resources.
- `docker-compose.override.yml` has developer specific overrides, and may be useful to customize the how the docker-compose runs on your machine. For example, it's useful to define hardware resource limits such as [CPU usage](https://docs.docker.com/compose/compose-file/deploy/#cpus) on the docker containers.
- `docker-compose-cicd.yml` is the one used during the CI pipelines defined as github actions in `.github/workflows/`
[^1]: Mais informações sobre o ray [aqui](https://docs.ray.io/en/releases-2.4.0/).

## Automated Tests

Now we describe how to run different kinds of tests that exist in this repository.

### Backend Tests

Refer to [pytest documentation](https://docs.pytest.org/en/7.2.x/) to know how to best select what backend tests to run.

First, start the necessary docker-compose services. Integration tests require all core services to be working, and unit tests require only the backend and databases.

Here are different test commands that are used frequently:

- Run all tests:

    ```bash
    docker compose exec backend pytest
    ```

- Run tests with HTML coverage reports (placed on `cov/` folder). The report ranks source files by line coverage, i.e. the relative number of (useful) lines that are passed at least once during the automated test execution. This script accepts the same arguments as coverage.py python test run command:

    ```bash
    docker compose exec backend bash scripts/test.sh pytest
    ```

- Run integration tests:

    ```bash
    docker compose exec backend pytest -m 'integration'
    ```

    To add a test in the integration group, decorate the test function with `@pytest.mark.integration`.

- Run unit tests:

    ```bash
    docker compose exec backend pytest -m 'not integration'
    ```

    Tests that are not integration falls into the unit test category by default.

Some tests will fail if ran outside docker-compose. Refer to the defined pytest marks to see what
kinds of test are available. We classify integration tests as those that use the backend running stack to test a function, and unit tests those tests that are self contained (or dependent of the databases only).

### Frontend and Cypress Tests

There are different test engines in the `webapp/` folder:

1. The jest unit tests:

    ```bash
    npm run test:unit
    ```

2. The Cypress component tests:

    ```bash
    # headless mode (now browser is launched to visualize the test execution)
    npm run cypress:run:component
    ```

    or

    ```bash
    # browser mode
    npm run cypress:open:component
    ```

3. The Cypress e2e tests:

    ```bash
    # headless mode
    npm run cypress:run
    ```

    or

    ```bash
    # browser mode
    npm run cypress:open
    ```


### Backend development

For altering the backend and take advantage of developer tools, you'll want to start a virtual environment with poetry. This will keep all the dependencies listed in `backend/pyproject.toml` isolated from the tools you have on you're machine. You can do it with the following command:

```bash
poetry install
poetry shell
```

#### Live development with Python Jupyter Notebooks

To run jupyterlab withing the docker compose networks (in order to access services such as the db and ray), run:

```bash
docker compose exec backend bash # get a shell in container
$JUPYTER # use JUPYTER variable defined in docker-compose.override.yml
```

If calling `$JUPYTER` fails, possible causes are:

1. Docker-compose.override.yml was not used, in such case you can run in the backend container shell:

```bash
jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
```

2. Development dependencies were not installed, in such case you should
   get a shell in the container and run `poetry add jupyter`. The default
   development image (`dockerfiles/Dockerfile.cpu`) already has jupyter as a devekionebt dependency.

#### Migration

Migrations is a pattern for versioning database schemas. Each migration represents a change in the schema. The seed migration states the first version, and all other migrations have a parent migration that it depends on. For more information, refer to the tutorial of [`alembic`](https://alembic.sqlalchemy.org/en/latest/tutorial.html), the tool we use to manage migrations.

There are 3 make commands that many help with migrations, their names are self-explanatory:

1. `make migrate-backend`: Runs backend migrations
1. `make migrate-mlflow`: Runs mlflow migrations
1. `make migrate`: Runs all migrations

As during local development your app directory is mounted as a volume inside the container, you can also run the migrations with `alembic` commands inside the container and the migration code will be in your app directory (instead of being only inside the container). So you can add it to your git repository.

Make sure you create a "revision" of your models and that you "upgrade" your database with that revision every time you change them. As this is what will update the tables in your database. Otherwise, your application will have errors.

- Start an interactive session in the backend container:

```bash
$ docker compose exec backend bash
```

- If you created a new model in `./mariner/entities`, make sure to import it in `mariner/entities/__init__.py`, that Python module that imports all the models will be used by Alembic.

- After changing a model (for example, adding a column), inside the container, create a revision, e.g.:

```bash
$ alembic revision --autogenerate -m "Add column last_name to User model"
```

- Commit to the git repository the files generated in the alembic directory.

- After creating the revision, run the migration in the database (this is what will actually change the database):

```bash
$ alembic upgrade head
```

#### Running Benchmarks

When implementing a benchmark, place it along other tests, and name your test function as `test_benchmark_name-of-benchmarking-target`, and mark it with `@pytest.mark.benchmark`

To execute a set of benchmarks, use the following pytest options:

- `-s` pytest option to capture the test stdout. This is needed to see the time results of the benchmark.
- `-m benchmark` otherwise the benchmarks collected will be deselected by the pytest.ini `adopts`.

**Observations**:
1. There's currently only one benchmark, so don't get attached to this strategy of storing the benchmarks.


#### Publishing Release Notifications

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

## Monitoring and Observability

We have a `grafana` and `prometheus` services defined in `docker-compose.yml` that must be up to monitor the application.
Specifically, we monitor ray using their exported dashboards through a volume on ray-head `/tmp/ray/metrics/`
as documented [here](https://docs.ray.io/en/latest/ray-observability/ray-metrics.html).

The grafana credentials is `admin` and `123456` when running it thorugh docker compose.

Other metrics are yet to be exposed to prometheus so they can be visualized in grafana.

## Troubleshooting

Here we list problems that may happen during development and how to address it.

- Getting pre-commit failures because of style when trying to commit

  Try running the following command to run automatic formatters:

  ```bash
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
