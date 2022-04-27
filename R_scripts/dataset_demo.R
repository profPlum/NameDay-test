library(tidyverse)
library(here)
source('R_scripts/emb_funcs.R')

vocab_data = load_vocab_emb('./logs/com/vocab_emb_com.txt', './logs/com/lexicon_com.csv',
                            vocab_only=F, english_only=T)
vocab_emb = vocab_data$vocab_emb
interp_vocab_emb = vocab_emb %>% hard_max_emb_interp()

data = read_csv('./Data/efw_cc.csv') %>% na.exclude()
summary(data)

# select only columns which are known members of more abstract columns
verbose_cols = grep('\\d[a-z]_', names(data))
verbose_data = data[,verbose_cols]
abstract_cols = grep('\\d_', names(data))
abstract_data = data[,abstract_cols]

# change names to a format that our algorithm can use
names(verbose_data) = gsub('\\d[a-z]?_', '', names(verbose_data))
names(verbose_data) = gsub('_', ' ', names(verbose_data))

head(verbose_data)

short_hand = list(marg='marginal', gov='government', std='standard', ppl='people', reg='regulation')

simplify_df = function(vocab_emb, df, rank) {
  vocab = rownames(vocab_emb)
  PCA = prcomp(df, rank=rank, scale=T)
  
  # df = t(one_hots), PCA$rotation = t(weights)
  # that is why there is left multiplication
  stopifnot(all.equal(scale(df) %*% PCA$rotation, PCA$x))
  
  # we can treat this just like a word embedding problem
  # the reason that 
  col_emb = PCA$rotation %>% norm_emb()
  interp_col_emb = col_emb %>% hard_max_emb_interp()
  compare_interp_scores(interp_col_emb)
  
  heatmap(interp_col_emb)
  
  colname_vocab = colnames(df) %>% strsplit(' ') %>% map(~if_else(.x %in% names(short_hand), short_hand[.x], as.list(.x))) %>%
    map(as_vector)
  
  # make DTM from column names
  # assumes no repeat words in colnames!
  colname_dtm = matrix(0, ncol(df), nrow(vocab_emb))
  colnames(colname_dtm) = rownames(vocab_emb)
  in_vocab_mask = colname_vocab %>% map(~all(.x %in% vocab)) %>% as_vector()
  if (!all(in_vocab_mask)) warning(c('not in vocab: ', colnames(df)[!in_vocab_mask]))
  colname_vocab %>% map(~table(.x)) %>% walk2(1:nrow(colname_dtm), ~{colname_dtm[.y,names(.x)]=.x})
   
  # same format as vocab embeddings
  colname_emb = colname_dtm %*% vocab_emb
  rownames(colname_emb) = colnames(df)

  # PC_names = get_PC_names(interp_col_emb)
  # PC_names = t(t(colname_emb) %*% interp_col_emb %*% PC_names)
  # rownames(PC_names) = embed(PC_names, vocab_emb, reverse=T)
  
  colname_vocab = colname_vocab %>% as_vector() %>% unique()

  # here colname_emb takes place of vocab_emb
  # and t(PCA$rotation)=W
  #vocab_emb = vocab_emb[rownames(vocab_emb) %in% colname_vocab,]
  PC_names = get_PC_names(colname_emb, vocab_emb = vocab_emb,
                          emb_weights=t(interp_col_emb))
  
  colnames(interp_col_emb) = rownames(PC_names)
  View(get_maximal_examples(interp_col_emb))
  
  # VERIFIED TO WORK
  simple_df = scale(df) %*% interp_col_emb
  colnames(simple_df) = rownames(PC_names)
  return(as_tibble(simple_df))
}

(simple_df = simplify_df(vocab_emb, verbose_data, rank=5))

# TODO: make this more expressive
main_PCs = simple_df[,1:2]
plot(main_PCs[[1]], main_PCs[[2]], main='simple PC plot',
     xlab=names(main_PCs)[[1]], ylab=names(main_PCs)[[2]])

cor(simple_df, abstract_data)

data = read_csv('~/Downloads/housing.csv') %>% clean_data()
simple_df = simplify_df(vocab_emb, data, rank=5)
