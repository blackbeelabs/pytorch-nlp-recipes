# Pytorch NLP Recipes

## Introduction
Pytorch NLP Recipes is a simple implementation of a deep learning project for NLP, in particular PyTorch. This project contains two problem statements:

- Sentiment Analysis
- Named Entity Recognition

The workflow contains a couple of steps:
1. Ingestion to a datamart (Schema used for training & testing)
2. Building a model
3. Evaluation of a model
4. Deployment (used for inference)

## Quickstart
Initialise your Python environment. I use `pyenv` / `virtualenv`, and 3.11.1
```
pyenv virtualenv 3.11.1 pytorchstuff
pyenv activate pytorchstuff
pip install -r requirements.txt
```

Set the `PYTHONPATH` and go to the `src` folder
```
export PYTHONPATH="/path/to/pytorch-nlp-recipes/src"
cd /path/to/pytorch-nlp-recipes/src
```

Run `build.py` and `train.py` for Sentiment Analysis
```
python pipelines/sentiment_analysis/ml/build.py
python pipelines/sentiment_analysis/ml/train.py
```

Observe that the models are saved in:
```
├── assets
│   ├── config
│   ├── datasets
│   │   └── sentiment_analysis
│   │       └── model
│   │           ├── vocabulary-build-baseline-20240308T103352.csv
│   │           ├── vocabulary-model-baseline-20240308T104310.csv
│   ├── models
│   │   ├── named_entity_recognition
│   │   └── sentiment_analysis
│   │       ├── build-baseline-20240308T103352.pkl
│   │       ├── buildprofile-baseline-20240308T103352.json
│   │       ├── model-baseline-20240308T104310.pkl
│   │       └── modelprofile-baseline-20240308T104310.json
```

Run `evaluate.py` via
```
python pipelines/sentiment_analysis/eval/evaluate.py -t "20240308T104310" # Update the timestamp
```

Observe that the reports are in:
```
├── models
│   ├── named_entity_recognition
│   └── sentiment_analysis
│       ├── detailedreport-model-baseline-20240308T104310.csv
│       └── report-model-baseline-20240308T104310.json
```

Repeat for Named Entity Recognition:
```
python pipelines/named_entity_recognition/pipelines/ml/build.py
python pipelines/named_entity_recognition/pipelines/ml/train.py
python pipelines/named_entity_recognition/pipelines/eval/evaluate.py -t 20240308T111229 # Update this
```

## Folder Structure

Some of the key features of this folder structure are:
1. Have datasets to progress from `downloaded`, `datamart` and finally `model`
2. Have all pipelines to be separated per project
3. Have `utils` and `model` code in separate packages independent of project
```
.
├── README.md
├── assets
│   ├── config
│   │   ├── named_entity_recognition
│   │   │   └── config-baseline.yaml
│   │   └── sentiment_analysis
│   │       ├── config-baseline.yaml
│   │       └── config-delta.yaml
│   ├── datasets
│   │   ├── named_entity_recognition
│   │   │   ├── datamart
│   │   │   │   └── labels.csv
│   │   │   ├── downloaded
│   │   │   │   └── source.md
│   │   │   └── model
│   │   │       ├── test-baseline.csv
│   │   │       ├── train-baseline.csv
│   │   │       └── vocabulary-baseline.csv
│   │   └── sentiment_analysis
│   │       ├── datamart
│   │       │   ├── test.csv
│   │       │   └── train.csv
│   │       ├── downloaded
│   │       │   └── source.md
│   │       └── model
│   ├── models
│   │   ├── named_entity_recognition
│   │   └── sentiment_analysis
│   └── notebooks
├── requirements-jupyter.txt
├── requirements.txt
└── src
    ├── model
    │   └── model.py
    ├── pipelines
    │   ├── named_entity_recognition
    │   │   └── pipelines
    │   │       ├── deploy
    │   │       │   └── expose_model.py
    │   │       ├── eval
    │   │       │   ├── evaluate.py
    │   │       │   └── model_architecture.py
    │   │       ├── ml
    │   │       │   ├── build.py
    │   │       │   ├── prepare_data.py
    │   │       │   ├── qa_data_loader.py
    │   │       │   └── train.py
    │   │       └── preprocess
    │   │           └── ingest_to_datamart.py
    │   └── sentiment_analysis
    │       ├── eval
    │       │   └── evaluate.py
    │       ├── ml
    │       │   ├── build.py
    │       │   └── train.py
    │       └── preprocess
    │           └── ingest_to_datamart.py
    ├── tasks
    │   └── tasks.py
    └── utils
        ├── io_utils.py
        └── text_utils.py
```
