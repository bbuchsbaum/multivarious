# Check if preprocessor is fitted and error if not

Internal helper to provide consistent error messages when attempting to
transform with unfitted preprocessors.

## Usage

``` r
check_fitted(object, action = "transform")
```

## Arguments

- object:

  A preprocessing object

- action:

  Character string describing the attempted action
