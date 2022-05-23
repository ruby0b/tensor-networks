test:
	pipenv run python -m pytest tests

lint:
	-flake8
	find examples/ tensor_networks/ -type f -name "*.py" | xargs mypy
