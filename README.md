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

## Setup environment
Initialise your Python environment. I use `pyenv` / `virtualenv`, and 3.11.1
```
make setup
```

## Ingest datasets
```
make ingest
```

## Train
```
make train
```

## Eval
```
make eval
```

## Remove environment
```
make remove
```

