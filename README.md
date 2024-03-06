# Pytorch NLP Recipes

A simple implementation of a sentiment analysis task with PyTorch.

To prepare:
1. Install requirements with `pip install -r requirements.txt`
2. Go to `path/to/src`
3. Run `pipelines/ml/build.py` to ensure that the model can be built
4. Run `pipelines/ml/eval.py` to validate that the model can be used for inference. Change the variables `workflow`, `experiment` and `experiment_timestamp` to ensure they are consistent with the output.
5. If all runs well, run `python pipelines/ml/train.py` and `python pipelines/ml/eval.py` to train your model. Observe how the performance improves between the built model with randomly initialised weights, and the trained model.