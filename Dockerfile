FROM python:3.9

WORKDIR /app/

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./app/pyproject.toml ./app/poetry.lock /app/


RUN /usr/local/bin/python3 -m pip install --upgrade pip

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install Poetry
RUN pip install -U pip setuptools \
    pip install poetry
ENV PATH="${PATH}:${POETRY_VENV}/bin"
RUN poetry config virtualenvs.create false
# Allow installing dev dependencies to run tests
# RUN ["poetry", "install", "--no-root"]
RUN poetry install --no-root
# Using inside the container:
# jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
RUN bash -c "pip install jupyterlab"
RUN apt-get install libpq-dev -y
COPY ./app /app
RUN bash -c "poetry run install_deps_${GEO_DEPS-cpu}"
ENV PYTHONPATH=/app
COPY --from=quay.io/oauth2-proxy/oauth2-proxy:v7.3.0-amd64 /bin/oauth2-proxy /usr/local/bin/oauth2-proxy

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
# CMD [ "sh", "-c", "poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 80"]
COPY --from=quay.io/oauth2-proxy/oauth2-proxy:v7.3.0-amd64 /bin/oauth2-proxy /usr/local/bin/oauth2-proxy
COPY ./entrypoints/mariner.sh ./
RUN chmod +x ./mariner.sh
ENTRYPOINT ["/app/mariner.sh"]