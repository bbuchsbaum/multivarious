# Enhanced fitted state tracking

Adds a fitted flag to preprocessing objects to track their state. This
is used by the new API to ensure proper workflow.

## Usage

``` r
mark_fitted(object, fitted = TRUE)
```

## Arguments

- object:

  A preprocessing object

- fitted:

  Logical indicating fitted state

## Value

The object with fitted state marked
