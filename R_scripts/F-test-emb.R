library(tidyverse)
library(glmnet)

vocab_emb = read.table('vocab_emb.txt')

models = NULL
lasso_models = NULL

# downsize
#sample_ids = sample(nrow(vocab_emb), 10)
vocab_emb = tail(vocab_emb, 100)

for (col in vocab_emb) {
  models = c(models, list(lm(col ~ diag(length(col)))))
  lasso_models = c(lasso_models, list(glmnet(diag(length(col)), col)))
  print("lasso model:")
  print(lasso_models[length(models)])
  summary(lasso_models[length(models)])
}

