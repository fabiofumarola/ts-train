# How run the repository

## Environment
Create a new conda environment with the following command:
```sh
conda create --name env_name python=3.10
conda activate env_name
```

Then install the poetry package inside the environment with:

```sh
pip install poetry
```

and then install all the requirements with:
```sh
poetry install
```

Now you can run the repository with:
```sh
python src/ts_train/main.py
```
---


## Pre-commit

Pre-commit hooks run all the auto-formatters (`black`), type checker (`mypy`), and a linter (`ruff`). These tools are used to be sure that the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---
## Makefile commands
The Makefile is a file used in software development projects to automate various tasks and streamline the build process. To use the Makefile, simply run the make command followed by the name of the target or rule specified in the Makefile.

In our new repository, we have included a special command in the Makefile called `check` that allows you to run all the steps of the CI/CD pipeline offline. This command is designed to save time by executing the necessary tasks locally before pushing the code. By running make `check`, you can perform all the necessary checks without triggering the entire CI/CD pipeline.

You can using before every push by calling:
```sh
make check
```

### Dockerfile
There are two commands into the Makefile thats help with dockerfile interaction:
```sh
make docker_build
```
this command generate the requirements.txt and then build the dockerfile```

```sh
make explore_docker
```
this command allow you to run the docker container with interaction mode.