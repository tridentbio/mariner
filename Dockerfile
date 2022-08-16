FROM python:3.9

WORKDIR /app/

# Copy poetry.lock* in case it doesn't exist in the repo
COPY ./app/pyproject.toml ./app/poetry.lock /app/


RUN /usr/local/bin/python3 -m pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false
# Allow installing dev dependencies to run tests
RUN ["poetry", "install", "--no-root"]
# Using inside the container:
# jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
RUN bash -c "pip install jupyterlab"
RUN apt-get install libpq-dev -y
COPY ./app /app
RUN bash -c "poetry run install_deps_${GEO_DEPS-cpu}"
ENV PYTHONPATH=/app

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD [ "sh", "prestart.sh", "&&", "sh", "-c", "poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 80"]
