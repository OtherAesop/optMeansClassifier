# optMeansClassifier
A K-means classifier built for the OptDigits data set

## Abstract

This is an experiment to examine the effects of k-means clustering on simple data and
to examine the resulting visualizations. We do this by controlling for the sensitivity to 
random initialization by doing 5 training runs on random integer seeds between 0 - 1000 and picking
the run with the lowest Mean Square Error. We then train on the best seed and attempt to classify
the training data with the resulting clusterings to examine our effectiveness.

We also visualize our trained clusters to see what group they most closely correspond to and label them
with the following rule.

`cluster_img + cluster_index + - + associated label + k + total number of clusters`

Because there are k images and 10 possible labels for each we expect that running this experiment will
make a maximum of 10 * k uniquely named visualization files.

## Experiments

The automation suite is designed to run the experiment described above for k of any value. Here we run
the experiment for k = 10 and k = 30.

If the first `verbose` flag is set to `True` then progress and output will be printed to the console

If the second `save` flag is set to `True` then the automation suite will save test results to the test
directory.

## How to use

Clone to your development environment of choice and run `main.py` with any Python 3.7 interpreter

