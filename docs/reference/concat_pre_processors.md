# bind together blockwise pre-processors

concatenate a sequence of pre-processors, each applied to a block of
data.

## Usage

``` r
concat_pre_processors(preprocs, block_indices)
```

## Arguments

- preprocs:

  a list of initialized `pre_processor` objects

- block_indices:

  a list of integer vectors specifying the global column indices for
  each block

## Value

a new `pre_processor` object that applies the correct transformations
blockwise

## Examples

``` r
p1 <- prep(center())
p2 <- prep(center())

x1 <- rbind(1:10, 2:11)
x2 <- rbind(1:10, 2:11)

p1a <- init_transform(p1,x1)
p2a <- init_transform(p2,x2)

clist <- concat_pre_processors(list(p1,p2), list(1:10, 11:20))
t1 <- apply_transform(clist, cbind(x1,x2))

t2 <- apply_transform(clist, cbind(x1,x2[,1:5]), colind=1:15)
```
