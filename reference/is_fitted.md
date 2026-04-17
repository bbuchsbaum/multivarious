# Check if a preprocessing object is fitted

Determine whether a preprocessing object has been fitted to data. This
is used internally to provide helpful error messages when users try to
transform data with an unfitted preprocessor.

## Usage

``` r
is_fitted(object)
```

## Arguments

- object:

  A preprocessing object to check

## Value

Logical: TRUE if fitted, FALSE otherwise
