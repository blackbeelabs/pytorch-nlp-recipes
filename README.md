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

## Folder Structure
```
├── config
├── datasets
│   ├── (project)
│   │   ├── datamart
│   │   │   ├── test.csv
│   │   │   └── train.csv
│   │   ├── downloaded
│   │   │   └── source.md
│   │   └── model
│       └── model
├── models
│   ├── (project)
└── notebooks
```
The assets
To prepare:
1. Install requirements with `pip install -r requirements.txt`
2. Go to `path/to/src`
3. Run `pipelines/ml/build.py` to ensure that the model can be built
4. Run `pipelines/ml/eval.py` to validate that the model can be used for inference. Change the variables `workflow`, `experiment` and `experiment_timestamp` to ensure they are consistent with the output.
5. If all runs well, run `python pipelines/ml/train.py` and `python pipelines/ml/eval.py` to train your model. Observe how the performance improves between the built model with randomly initialised weights, and the trained model.