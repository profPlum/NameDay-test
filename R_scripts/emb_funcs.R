.script.dir <- dirname(sys.frame(1)$ofile)
setwd(.script.dir)
library(tm)
library(tokenizers)
library(tidyverse)
library(qdapDictionaries)
library(moments)

generate_part_sets = function(vocab_emb, n_parts=5, exclude_PC_names=T,
                              low_freq_only=F) {
  vocab = rownames(vocab_emb)
  weights = t(vocab_emb)
  
  if (exclude_PC_names) {
    excluded_vocab = stemDocument(vocab) %in% stemDocument(colnames(vocab_emb))
    weights[, excluded_vocab] = -Inf
  }
  
  if ('abs_scores' %in% names(attributes(vocab_emb))) {
    heatmap(vocab_emb)
    important_parts_idx = apply(vocab_emb, -1, FUN=sort, index.return=T, decreasing=T) %>% map(~.x$ix)
    
    # weight vocab_emb by abstractness scores s.t. less abstract words are chosen as parts
    vocab_emb = vocab_emb %>% scale %>% sweep(1, attr(vocab_emb, 'abs_scores'))
    
    # sort offset plots (how did abs scores change ordering?)
    vocab_emb_sort_offset = important_parts_idx %>% map2(1:dim(vocab_emb)[2], ~vocab_emb[,.y][.x]) %>%
      reduce(cbind)
    image(vocab_emb_sort_offset)
    heatmap(vocab_emb)
    hist(attr(vocab_emb, 'abs_scores'))
  }
  
  # we want to reduce the columns dimensions (dim=2) & sort int is like np.argsort()
  important_parts_idx = apply(weights, -2, FUN=sort, index.return=T, decreasing=T) %>% map(~.x$ix)
  
  # clean parts:
  clean = 1:length(vocab)  # until told otherwise everything is 'clean'
  if (low_freq_only) {
    # why *2?
    dtm = important_parts_idx %>% map(~{r=rep(0, length(vocab)); r[.x[1:n_parts*2]]=1; r}) %>% reduce(rbind)
    freqs = dtm %>% apply(-1, FUN=sum)  # word counts
    low_freq = which(freqs <= (mean(freqs)+sd(freqs)))
    # ^ anything greater than 1 *standard deviation* above the mean is too high (hence the name)
    
    clean = intersect(clean, low_freq)
  }
  
  important_parts = important_parts_idx %>% map(~intersect(.x, clean)) %>%
    map(~vocab[.x[1:n_parts]])
  
  return(important_parts)
}

hard_max_emb_interp = function(vocab_emb) {
  greedy_PC_name_selection = function(vocab_emb) {
    # we set the first member to be the vocab vector with the highest
    # PC dimension of all (and thereby most aligned with its axis)
    target_PC_names = apply(vocab_emb, -2, max) %>% which.max
    target_PC_names = as.matrix(vocab_emb[target_PC_names,])
    
    for (i in 2:dim(vocab_emb)[2]) {
      sim = vocab_emb %*% target_PC_names
      ortho_loss = apply(sim, -2, norm, type='2') # two norm prioritizes minizization of max similarity
      target_PC_names = cbind(target_PC_names, vocab_emb[which.min(ortho_loss),])
    }
    return(t(target_PC_names)) # we want standard row format
  }
  greedy = greedy_PC_name_selection(vocab_emb)
  PC_names = t(vocab_emb)%*%vocab_emb
  
  # ((V^TVG^-1)^-1)^T = Q
  # V^TV = (Q^T)^-1G
  Q = t(solve(PC_names %*% solve(greedy)))
  vocab_emb = vocab_emb %*% Q
  colnames(vocab_emb) = rownames(greedy)
  PC_names = t(vocab_emb)%*%vocab_emb
  stopifnot(all.equal(PC_names, greedy%*%Q))
  return(norm_emb(vocab_emb))
}

cos_sim_diff = function(V, O, sample_sz=200) {
  cos_sim_plot = function(V, n_plot, title=NULL) {
    cos_sims_V = V%*%t(V)
    cos_sims_plot = cos_sims_V[1:n_plot, 1:n_plot]
    coors = cmdscale(-cos_sims_plot-min(-cos_sims_plot))
    axis_lim = 0.8
    p = ggplot(as_tibble(coors), aes(x=V1, y=V2)) + geom_point(aes(color='yellow', alpha=0.3)) +
      geom_text(aes(label=rownames(coors)), position = 'dodge', check_overlap = T) + ggtitle(title) +
      theme(legend.position = "none") + xlim(-axis_lim, axis_lim) + ylim(-axis_lim, axis_lim)
    print(p)
    return(cos_sims_V)
  }
  
  ids = sample(dim(V)[1], sample_sz)
  O = norm_emb(O[ids,])
  V = V[ids,]
  
  n_plot = 10
  cos_sims_V = cos_sim_plot(V, n_plot)
  cos_sims_O <- cos_sim_plot(O, n_plot)
  
  cos_sim_differences = cos_sims_O - cos_sims_V
  dim(cos_sim_differences) = NULL
  ##hist(cos_sim_differences)
  boxplot(cos_sim_differences)
  return(mean(abs(cos_sim_differences)))
}

topn_similar = function(vocab_emb, words, n=10) {
  if (is(words, 'character')) {
    words = embed(words, vocab_emb)
  }
  if (is.null(dim(words))) {
    words = t(matrix(words))
  }
  
  top_ids = apply(vocab_emb %*% t(words), -1, sort, decreasing=T, index.return=T) %>% map('ix')
  if (is.null(dim(top_ids))) {
    top_ids = top_ids[1:n]
  } else {
    top_ids = top_ids[1:n,]
  }
  return(rownames(vocab_emb)[as_vector(top_ids)])
}

image.real = function(mat, main=NULL) image(t(mat)[,nrow(mat):1], main=main)

compare_interp_scores = function(vocab_emb,  dtm=NULL) {
  Zobnin_interpretability = vocab_emb %>% get_PC_interp_scores()
  Standard_interpretability = standard_interp_scores(vocab_emb, dtm=dtm)
  cat('Mean Zobnin score: ', mean(Zobnin_interpretability), 'Mean Standard score:', mean(Standard_interpretability), '\n')
  cat('Median Zobnin score: ', median(Zobnin_interpretability), 'Median Standard score:', median(Standard_interpretability), '\n')
  # boxplot(russ_interp, main='Zobnin Interpretability')
  # boxplot(std_interp, main='Standard Interpretability')
  cat("Score correlation: ", cor(Zobnin_interpretability, Standard_interpretability), '\n')
  return(invisible(lst(Zobnin_interpretability, Standard_interpretability)))
}
add_abstractness_scores = function(vocab_emb,
                                   sim_mat = vocab_emb%*%t(vocab_emb)) {
  abs_scores = scale(apply(sim_mat, -1, skewness))
  names(abs_scores) = rownames(vocab_emb)
  attr(vocab_emb, 'abs_scores') = abs_scores
  return(vocab_emb)
}

# get different term-cooccurence-matrix types
.term_mat_choices = c('dtm','tcm','pmi','ppmi')
get_PCA = function(dtm, rank=NULL, type=.term_mat_choices) {
  if (type[1]!='dtm') tcm = t(dtm)%*%dtm
  term_mat = switch(which(type[1]==.term_mat_choices),
                    dtm, tcm, get_PMI(tcm), get_PMI(tcm, positive=T))
  
  # we need to scale and center because otherwise PCA doesn't work well
  PCA = prcomp(term_mat, scale=T)
  return(PCA)
}

simplest_forms = function(vocab_emb) {
  stemmed_vocab = stemDocument(rownames(vocab_emb))
  for (word_stem in unique(stemmed_vocab)) {
    forms = unique(rownames(vocab_emb)[word_stem == stemmed_vocab])
    simplest_form_i = which.min(nchar(forms))
    
    redundant_vocab_i = which(rownames(vocab_emb) %in% forms[-simplest_form_i])
    if (length(redundant_vocab_i)>0)
      vocab_emb = vocab_emb[-redundant_vocab_i,]
  }
  return(vocab_emb)
}

load_corpus = function(corpus_fn, lexicon_fn, n_grams=-1L, gram_sz = 5, n_PCs = 10,
                       extra_stop_words = NULL, term_mat_type=.term_mat_choices, stem=F, use_cache=F) {
  clean_text <- function(text)
    text %>% tolower %>% removePunctuation %>% removeNumbers %>% removeWords(c(extra_stop_words,stopwords('en'))) %>% 
    stripWhitespace # stemming done elsewhere
  get_dtm = function(docs)
    docs %>% VectorSource %>% VCorpus %>% DocumentTermMatrix
  
  raw_corpus = paste(readr::read_lines(corpus_fn, n_max=n_grams),collapse='\n') %>%
    clean_text %>% chunk_text(chunk_size=gram_sz)
  if (n_grams!=-1) raw_corpus = sample(raw_corpus, n_grams)
  print(head(raw_corpus))
  
  corpus = clean_text(raw_corpus)
  if (stem) corpus = stemDocument(corpus)
  dtm = as.matrix(get_dtm(corpus))
  PCA = get_PCA(dtm, term_mat_type)
  
  lexicon = read_csv(lexicon_fn)
  if (stem) lexicon = lexicon %>% mutate(Word=stemDocument(Word))
  vocab_emb = get_vocab_emb(PCA, n_PCs, term_mat_type)
  return(lst(vocab_emb, lexicon, dtm))
}

load_vocab_emb = function(emb_fn, lexicon_fn, 
                          sample_size=Inf, vocab_only=F, english_only=F, defined_only=F) {
  vocab_emb = as.matrix(read_table2(emb_fn, col_names=F))
  #stopifnot(all.equal(norm(as.matrix(vocab_emb[1,]), type='2'), 1))
  
  mask = T
  if (vocab_only) {
    lexicon = as_factor(as_vector(read_table(lexicon_fn, col_names = F)))
    rownames(vocab_emb) = lexicon
  } else {
    lexicon = read_csv(lexicon_fn)
    rownames(vocab_emb) = as_factor(as_vector(lexicon$Word))
    # we want this attached to vocab_emb so we can sample easily
    mask = (!defined_only | rownames(vocab_emb) %in% lexicon[!is.na(lexicon$Definition),]$Word)
  }
  
  mask = mask & (!english_only | rownames(vocab_emb) %in% GradyAugmented)
  
  lexicon = switch(is_tibble(lexicon)+1, lexicon[mask], lexicon[mask,])
  vocab_emb = vocab_emb[mask,]
  
  sample_ids = sample(nrow(vocab_emb), min(sample_size, nrow(vocab_emb)))
  vocab_emb = vocab_emb[sample_ids,]
  return(tibble::lst(vocab_emb, lexicon))
}

# verified to work!
# PMI(w,c) = log(P(w,c)/(P(w)*P(c))) The true probability of coocurrence, 
# over the probability of cooccurrence if they were indep. 
# Then within a log() since *it is a ratio* & it makes the baseline 0
get_PMI = function(TCM, positive=T) {
  TCM = TCM - min(TCM) # occurrences cannot be negative
  
  total = sum(TCM)
  P_wc = TCM/total # P(w,c) word context pair
  P_w = apply(TCM,-2,sum)/total # P(w) word alone
  P_c = apply(TCM,-1,sum)/total # P(c) context alone
  PMI = log(P_wc/(P_w%o%P_c)) # log has special properties for info theory
  if (positive) PMI = pmax(PMI, 0)
  return(PMI)
}

# interp_scores made conventional in topic modeling
# :param sim_mat: measures similarity/cooccurrence between words (PMI is recommended)
standard_interp_scores = function(vocab_emb, n=5, dtm=NULL) {
  vocab = rownames(vocab_emb)
  stopifnot(!('abs_scores' %in% names(attributes(vocab_emb))))
  top_parts = generate_part_sets(vocab_emb, n, exclude_PC_names = F)
  
  # don't make full sim mat
  all_parts <- embed(as_vector(top_parts), vocab_emb)
  term_mat = switch(is.null(dtm)+1, t(dtm)%*%dtm, all_parts%*%t(all_parts))
  sim_mat=get_PMI(term_mat)
  
  # don't want self-sim interfering with calcs
  diag(sim_mat)=0
  n_word_pairs = (n**2-n)
  
  interp_scores = top_parts %>% map(~rownames(all_parts) %in% .x) %>%
    map(~sum(sim_mat[.x,.x])/n_word_pairs)
  # last map is taking the mean over square formed by .x,.x - diag()
  
  return(as_vector(interp_scores))
}

# accepts all embeddings in standard m x n_PC vocab_emb format
embed = function(words, vocab_emb, reverse=F, weight_abs=F) {
  if (reverse) {
    if (!is.null(dim(words))) words = t(words)
    word_sim = scale(vocab_emb %*% words)
    if (weight_abs) word_sim = sweep(word_sim, 1, -attr(vocab_emb, 'abs_scores'))
    PC_name_indices = apply(word_sim, MARGIN=-1, FUN=which.max)
    vocab = rownames(vocab_emb)
    return(vocab[PC_name_indices])
  } else {
    return(vocab_emb[which(rownames(vocab_emb) %in% words),])
  }
}

norm_emb = function(emb, type='2') {
  norms = apply(emb, -2, function(x) norm(as.matrix(x), type=type))
  return(emb / ifelse(norms>0, norms, 1))
}

get_vocab_emb = function(PCA, n_PCs, term_mat=.term_mat_choices, use_abs=F) {
  if (!use_abs) warning('abs is False for PCA!!!')
  
  # for all term mats other than dtm we can use PCA$x (since they are derived from TCM)
  vocab_emb = switch((term_mat[1]=='dtm')+1, PCA$x, PCA$rotation)
  vocab_emb = norm_emb(vocab_emb[,1:n_PCs])
  if (use_abs) vocab_emb = abs(vocab_emb)
  
  # if embs they are already set & not found on PCA$rotation rows...
  if (term_mat!='emb') {
    vocab = rownames(PCA$rotation)
    rownames(vocab_emb) = vocab
  }
  return(vocab_emb)
}

# to get in +- pass abs(vocab_emb)
get_maximal_examples = function(vocab_emb, use_abs=F) {
  vocab = rownames(vocab_emb)
  if (use_abs) vocab_emb = abs(vocab_emb)
  sort_ids = apply(vocab_emb, -1, sort, index.return=T) %>% map(~.x$ix)
  
  # tail because sort is increasing
  maximal_examples = sort_ids %>% map(~vocab[tail(.x)]) %>% 
    reduce(rbind) %>% t()
  colnames(maximal_examples) = colnames(vocab_emb)
  maximal_examples %>% View()
  return(maximal_examples)
}

# optimized diagonal of dot product
diag_dot = function(A, B) apply(A*t(B), -2, sum)

# NOTE: interestingly when vocab embeddings aren't normalized 
# then interpretability scores are all 1
get_PC_interp_scores = function(vocab_emb) {
  W=vocab_emb # use naming convention of paper
  
  # the real eq is interp_k = (W^TWW^TW)_k,k
  # but we optimize it a little
  PC_names = t(W)%*%W # this is by my own naming convention
  interp = diag_dot(PC_names, PC_names) # interp scores vector
  
  return(interp)
}

# this should maximize the interp of first PCs (at cost of later PCs)
max_PCs_interp = function(vocab_emb, method=c('qr','russ')) {
  if (method == 'qr') {
    Q = qr.Q(qr(vocab_emb))
    R = t(Q)%*%vocab_emb
    dimnames(R) = dimnames(vocab_emb)
    return(R)
  } else {
    W = vocab_emb # use naming convention of paper
    svd_ = svd(W)
    Q = svd_$v
    return(W%*%Q)
  }
}

# to get in +- pass abs(vocab_emb)
get_PC_names = function(in_emb,
                        vocab_emb=in_emb, emb_weights=t(in_emb), method=c('forward', 'max')) {
  if (method[1]=='max') {
    name_idx = apply(vocab_emb, -1, which.max)
    PC_names = vocab_emb[name_idx,]
    return(PC_names)
  }
  
  # we need to do a double transpose in order for weights to be treated as row-vec
  # weights are multiplied like this for the sake of accounting for eigen values
  PC_emb = emb_weights %*% in_emb # TODO: reconsider norming here again?
  weight_abs = 'abs_scores' %in% names(attributes(vocab_emb))
  
  rownames(PC_emb) = embed(PC_emb, vocab_emb, reverse=T, weight_abs=weight_abs)
  return(PC_emb)
}

