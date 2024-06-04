PYENV_VERSION=3.11.1
PROJECT_NAME=pytorch-nlp-recipes
BINARIES = $(HOME)/.pyenv/versions/${PROJECT_NAME}/bin

setup_env:
	pyenv virtualenv ${PYENV_VERSION} ${PROJECT_NAME}
	pyenv local ${PROJECT_NAME}
	pip install -r requirements.txt
	# pip install -r requirements-jupyter.txt
	
teardown_env:
	# pyenv deactivate ${PROJECT_NAME}
	pyenv uninstall ${PROJECT_NAME}

download_data:
	@echo "[Start]: download sentiment.csv"
	curl https://raw.githubusercontent.com/surge-ai/stock-sentiment/main/sentiment.csv > \
	./assets/datasets/sentiment_analysis/downloaded/sentiment.csv
	@echo "[Start]: download train.txt"
	curl https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/SEC-filings/CONLL-format/data/train/FIN5.txt > \
	./assets/datasets/named_entity_recognition/downloaded/train.txt
	@echo "[Start]: download test.txt"
	curl https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/SEC-filings/CONLL-format/data/test/FIN3.txt > \
	./assets/datasets/named_entity_recognition/downloaded/test.txt
	@echo "[Done]: download_data"
	
ingest_data:
	@echo "[Start]: ingest data for named entity recognition"
	${BINARIES}/python -m src.pipelines.named_entity_recognition.preprocess.ingest_to_datamart
	@echo "[Start]: ingest data for sentiment analysis"
	${BINARIES}/python -m src.pipelines.sentiment_analysis.preprocess.ingest_to_datamart
	@echo "[Done]: ingest_data"

