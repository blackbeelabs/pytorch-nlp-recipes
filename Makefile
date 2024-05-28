PYENV_VERSION=3.11.1
PROJECT_NAME=pytorch-nlp-recipes

setup_env:
	pyenv virtualenv ${PYENV_VERSION} ${PROJECT_NAME}
	pyenv local ${PROJECT_NAME}
	pip install -r requirements.txt
	pip install -r requirements-jupyter.txt
	
teardown_env:
	# pyenv deactivate ${PROJECT_NAME}
	pyenv uninstall ${PROJECT_NAME}
