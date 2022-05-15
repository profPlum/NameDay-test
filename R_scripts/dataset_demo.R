source('R_scripts/emb_funcs.R')
library(tidyverse)

clean_data = function(data) data %>% na.exclude() %>% select_if(is.numeric) %>% rename_all(~gsub('_',' ',.x))

vocab_data = load_vocab_emb('./logs/vocab_emb.txt', './logs/lexicon.csv',
                            vocab_only=F, english_only=T)
vocab_emb = vocab_data$vocab_emb
interp_vocab_emb = vocab_emb %>% hard_max_emb_interp()

data = read_csv('./Data/efw_cc.csv') %>% na.exclude()
cat('Summary of economic word freedom dataset (unsimplified):\n')
summary(data)

# select only columns which are known members of more abstract columns
verbose_cols = grep('\\d[a-z]_', names(data))
verbose_data = data[,verbose_cols]
abstract_cols = grep('\\d_', names(data))
abstract_data = data[,abstract_cols]

# change names to a format that our algorithm can use
names(verbose_data) = gsub('\\d[a-z]?_', '', names(verbose_data))
names(verbose_data) = gsub('_', ' ', names(verbose_data))

# short hand mapping to plain english
short_hand = list(marg='marginal', gov='government', std='standard', ppl='people', reg='regulation')

simplify_df = function(vocab_emb, df, rank, short_hand=list()) {
  vocab = rownames(vocab_emb)
  PCA = prcomp(df, rank=rank, scale=T)

  # df = t(one_hots), PCA$rotation = t(weights)
  # that is why there is left multiplication
  stopifnot(all.equal(scale(df) %*% PCA$rotation, PCA$x))

  # we can treat this just like a word embedding problem
  # the reason that
  col_emb = PCA$rotation %>% norm_emb()
  interp_col_emb = col_emb %>% hard_max_emb_interp()
  #compare_interp_scores(interp_col_emb)

  # make DTM from column names
  # assumes no repeat words in colnames!
  # expand shorthand
  colname_vocab = colnames(df) %>% strsplit(' ') %>%
    map(~if_else(.x %in% names(short_hand), short_hand[.x], as.list(.x))) %>%
    map(as_vector)
  
  # make DTM from column names
  # assumes no repeat words in colnames!
  colname_dtm = matrix(0, ncol(df), nrow(vocab_emb))
  colnames(colname_dtm) = rownames(vocab_emb)
  in_vocab_mask = colname_vocab %>% map(~all(.x %in% vocab)) %>% as_vector()
  if (!all(in_vocab_mask)) warning(c('not in vocab: ', colnames(df)[!in_vocab_mask]))
  
  # NOTE: global assignment operator is necessary! You should switch back from = assignment to <- assignment
  colname_vocab %>% map(table) %>% walk2(1:nrow(colname_dtm), ~{colname_dtm[.y,names(.x)] <<- .x})
  stopifnot(any(colname_dtm>0))
  # same format as vocab embeddings
  colname_emb = colname_dtm %*% vocab_emb

  # here colname_emb takes place of vocab_emb
  # and t(PCA$rotation)=W
  PC_names = get_PC_names(colname_emb, vocab_emb = vocab_emb,
                          emb_weights=t(interp_col_emb))

  colnames(interp_col_emb) = rownames(PC_names)
  heatmap(interp_col_emb, margins=c(7,7))
  View(get_maximal_examples(interp_col_emb))

  # VERIFIED TO WORK
  simple_df = scale(df) %*% interp_col_emb
  colnames(simple_df) = rownames(PC_names)
  return(as_tibble(simple_df))
}

cat('Summary of economic word freedom dataset (simplified):\n')
simple_df = simplify_df(interp_vocab_emb, verbose_data, rank=5, short_hand = short_hand)
summary(simple_df)

main_PCs = simple_df[,1:2]
plot(main_PCs[[1]], main_PCs[[2]], main='Simple PC Example Plot',
     xlab=names(main_PCs)[[1]], ylab=names(main_PCs)[[2]])

cat("Cross correlations of simplified dataframe and human engineered macro/summary variables:\n")
cor(simple_df, abstract_data)

