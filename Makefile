IMAGE_NAME := ts_train

docker_build:
	docker build -t $(IMAGE_NAME) .

docker_explore:
	docker run -it --rm --entrypoint=/bin/bash $(IMAGE_NAME)

check:

	black src
	black tests

	mypy src
	mypy tests

	ruff check src --fix
	ruff check tests --fix

	pytest tests -v -x --cov --cov-report html:coverage
