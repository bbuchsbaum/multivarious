# construct a random forest wrapper classifier

Given a model object (e.g. `projector` construct a random forest
classifier that can generate predictions for new data points.

## Usage

``` r
rf_classifier(x, colind, ...)
```

## Arguments

- x:

  the model object

- colind:

  the (optional) column indices used for prediction

- ...:

  extra arguments to `randomForest` function

## Value

a random forest classifier
