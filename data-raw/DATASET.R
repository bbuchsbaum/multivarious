## code to prepare `DATASET` dataset goes here

##usethis::use_data(DATASET, overwrite = TRUE)
devtools::load_all()
library(corrplot)
df1 <- read.csv("ProportionBlock_YA_OA_for_PCA.csv")
library(ggplot2)

mat <- as.matrix(df1[,c(2:4, 11:13,17:19)])
pres1 <- pca(mat, ncomp=9, preproc=center())

df2 <- data.frame(PC1 = pres1$s[,1], PC2 = pres1$s[,2], Age=factor(ifelse(df1$Age < 45, "young", "old")))
qplot(PC1, PC2, colour=Age, data=df2)

library(FactoMineR)
library(factoextra)
pres2 <- PCA(mat, scale=FALSE)
p <- fviz_pca_biplot(pres2, repel=TRUE, geom=c("point"), alpha=0, col.var="black") 
p + geom_point(aes(x=PC1, y=PC2, colour=Age, alpha=.2), data=df2)


library(MASS)

ld1 <- MASS::lda(mat, df2$Age)




