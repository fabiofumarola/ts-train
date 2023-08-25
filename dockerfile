FROM apache/spark-py:v3.4.0 as base
#This variable ensures that Python output is not buffered, allowing you to see the output in real-time.
ENV PYTHONUNBUFFERED=true
# Switch to root user
USER root
# Set the Python alias
RUN echo "alias python=python3" >> ~/.bashrc
# Switch back to non-root user
USER ${USER}
WORKDIR /ts_train


# create a separate repo with poetry installation, create the complete .env file here.
FROM base as poetry

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"

# copy the complete repo code
RUN pip install poetry
# install only the custom library without other requirements
COPY . ./
#RUN poetry install --no-interaction --no-ansi -vvv
RUN poetry export --without-hashes --format=requirements.txt > requirements.txt
RUN poetry build


FROM base as runtime
# take only requirements so the dockerfile do no see new edit in source code and do not invalide cache
COPY --from=poetry /ts_train/requirements.txt /ts_train/requirements.txt
RUN pip install -r requirements.txt

# get the .env from poetry image and use it without installation
COPY --from=poetry /ts_train /ts_train
RUN pip install dist/ts_train*.whl

#ENV PATH="/ts_train/.venv/bin:$PATH"
#CMD ["python", "user_code/test_docker.py"]
