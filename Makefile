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
	
preprocess_data:
	@echo "[Start]: ingest data for named entity recognition"
	${BINARIES}/python -m src.pipelines.named_entity_recognition.preprocess.ingest_to_datamart
	@echo "[Start]: ingest data for sentiment analysis"
	${BINARIES}/python -m src.pipelines.sentiment_analysis.preprocess.ingest_to_datamart
	@echo "[Done]: ingest_data"

train_sentiment_analysis_step:
	@echo "[Start]: training workflow for sentiment analysis"
	${BINARIES}/python -m src.pipelines.sentiment_analysis.ml.train
	@echo "[Done]: training for sentiment analysis"

train_named_entity_recognition_step:
	@echo "[Start]: training workflow for named entity recognition"
	${BINARIES}/python -m src.pipelines.named_entity_recognition.ml.prepare_data
	${BINARIES}/python -m src.pipelines.named_entity_recognition.ml.train
	@echo "[Done]: training for named entity recognition"
	
eval_named_entity_recognition_step:
	@echo "[Start]: eval workflow for named entity recognition"
	${BINARIES}/python -m src.pipelines.named_entity_recognition.ml.eval -t 20240604T172201
	@echo "[Done]: eval for named entity recognition"

eval_sentiment_analysis_step:
	@echo "[Start]: eval workflow for sentiment analysis" 
	${BINARIES}/python -m src.pipelines.sentiment_analysis.ml.eval -t 20240604T172156
	@echo "[Done]: eval for sentiment analysis"

setup: setup_env

ingest: download_data preprocess_data
train: train_sentiment_analysis_step train_named_entity_recognition_step
eval: eval_sentiment_analysis_step eval_named_entity_recognition_step

wf: ingest train

remove: teardown_env

