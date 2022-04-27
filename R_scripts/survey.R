library(tidyverse)
library(qdapDictionaries)
library(factoextra)
library(here)
source(here('R_scripts/emb_funcs.R'))

get_integrous_PC_order = function(important_part_idx) {
  # incoherence measures how far the original indices were pushed back in order to be valid.
  # Or in other words how distorted the valid representation of parts is relative to original.
  incoherence_scores = important_parts_idx %>% map(~sum(sort(which(.x %in% clean))[1:n_parts])/sum(1:n_parts))
  most_integrous_PC_indices = (integrity_scores %>% as_vector %>% sort(index.return=T))$ix
  
  # this measures the mean consime similarity between group members then it makes a barplot of it ordered by integrity
  # NOTE: currently there is no trend, this is potentially bad because it means that the coherence scores don't reflect group cohesion
  # that being said similarity on one dimension doesn't guarentee it on all dimensions
  important_parts_idx[most_integral_PC_indices] %>% map(~vocab_emb[.x[1:5],]) %>% map(~mean(.x%*%t(.x))-1/n_parts) %>%
    as_vector %>% barplot(main='group integrity vs cohesion', xlab='integrity', ylab='cohesion')
  # NOTE: -1/n_parts creates the unbiased mean (which is biased based on similarity of words with themselves)
  return(most_integral_PC_indices)
}

generate_part_sets = function(vocab_emb, n_parts=5, exclude_PC_names=T,
                              english_only=F, low_freq_only=F) {
  #stopifnot(!('PC1' %in% colnames(vocab_emb)))
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
  if (english_only) {
    is_english = vocab %in% GradyAugmented
    print(c('mean portion of words in english: ', mean(is_english)))
    is_english = which(is_english)
    clean = intersect(clean, is_english)
  }
  if (low_freq_only) {
    # why *2?
    dtm = important_parts_idx %>% map(~{r=rep(0, length(vocab)); r[.x[1:n_parts*2]]=1; r}) %>% reduce(rbind)
    freqs = dtm %>% apply(-1, FUN=sum)  # word counts
    low_freq = which(freqs <= (mean(freqs)+sd(freqs)))
    # ^ anything greater than 1 *standard deviation* above the mean is too high (hence the name)
    
    clean = intersect(clean, low_freq)
  }
  
  important_parts = important_parts_idx %>% map(~intersect(.x, clean)) %>% #walk(~print(c('remaining indices: ',.x))) %>% 
    map(~vocab[.x[1:n_parts]])
  
  
  # if (!is.null(names(important_parts))) {
  #   part_emb = important_parts %>% map2(names(important_parts), ~embed(c(.y, .x), vocab_emb))
  #   mean_group_sim = part_emb %>% map(~mean(.x%*%t(.x))) %>% as_vector() %>% as.numeric
  #   hist(mean_group_sim)
  #   
  #   part_emb %>% map(~mean(.x[-1,]%*%.x[1,])) %>% as_vector %>% hist(main='mean group to name sim')
  # }
  
  return(important_parts)
}

display_important_parts = function(important_parts, lexicon,  num=10, use_definitions=T, interactive=F) {
  sample_ids = sample(length(important_parts), min(num,length(important_parts)))
  if (!use_definitions) lexicon = lexicon$Word
  
  correct = NULL
  
  if (is_tibble(lexicon)) {
    safe_vocab = lexicon$Word
  } else {
    safe_vocab = lexicon
  }
  safe_vocab = setdiff(safe_vocab, names(important_parts))
  
  answer_key = NULL
  question_id = 1
  for (i in sample_ids) {
    cat(question_id,'. ')
    question_id = question_id + 1
    
    n = length(important_parts[[i]])
    
    # print (defined) words
    if (is.tibble(lexicon)) {
      stopifnot(all(as_vector(important_parts) %in% lexicon$Word))
      lexicon %>% filter(Word %in% important_parts[[i]]) %>% sample_n(n()) %>% print()
    } else cat('please name this set:', paste(sample(important_parts[[i]], n), collapse=', '), '\n')
    
    name = names(important_parts)[i]
    name_choices = c(sample(safe_vocab, n-1), name)
    
    if (is_tibble(lexicon)) {
      print('name choices:')
      stopifnot(all(as_vector(name_choices) %in% lexicon$Word))
      name_choices_def = lexicon %>% filter(Word %in% name_choices) %>% 
        sample_n(n())
      insert_pos = which(name_choices_def$Word==name)
      print(name_choices_def)
    } else {
      # put correct answer in random spot
      insert_pos = sample(n, 1)
      tmp = name_choices[insert_pos]
      name_choices[insert_pos] = name_choices[n]
      name_choices[n] = tmp
      cat(paste0(name_choices,'\n'))
    }
    
    if (interactive) {
      answer = readline('please choose (q=quit): ')
      if (answer=='q') break
      correct = c(correct, as.integer(answer)==insert_pos)
    }
    cat('\n', rep('-', 20), '\n\n')
    
    answer_key = c(answer_key, name_choices[insert_pos])
  }
  return(lst(correct, sample_ids, answer_key))
}

# Here we cluster embeddings then perform PCA on each cluster (individually)
# to simplify to embeddings in hopes that the resulting parts will be more 
# easily categorizable by humans.
do_name_survey = function(vocab_emb, lexicon=rownames(vocab_emb),
                          n_centers=10, n_questions=10, interactive=T, use_definitions=T) {
  # clusters = hclust(dist(vocab_emb[sample_ids,]))
  # plot(clusters)  # seems like 10-13 clusters is ideal
  
  # this takes forever cannot afford nstart>1
  clusters = kmeans(vocab_emb, centers=n_centers, nstart=1)
  cluster_ids = clusters$cluster
  
  rank = dim(vocab_emb)[2]/n_centers
  
  for (i in 1:n_centers) {
    mask = cluster_ids==i
    stopifnot(all(mask))
    vocab_emb_cluster = vocab_emb[mask,]
    attr(vocab_emb_cluster, 'abs_scores') = attr(vocab_emb, 'abs_scores')
    
    if (rank < dim(vocab_emb)[2]) {
      stop()
      PCA = prcomp(vocab_emb_cluster, rank=rank)
      reduced_embs = get_vocab_emb(PCA, rank, term_mat = 'emb')
      #plot(PCA)
    } else {
      reduced_embs = vocab_emb_cluster
    }
    
    vocab_cluster = rownames(vocab_emb_cluster)
    rownames(reduced_embs) = vocab_cluster
    # assign proper names to these embeddings
    
    PC_name_embs = get_PC_names(reduced_embs)
    colnames(reduced_embs) = rownames(PC_name_embs)
    
    results = generate_part_sets(reduced_embs) %>%
      display_important_parts(lexicon, num=n_questions,
                              interactive=interactive, use_definitions = use_definitions)
    correct = as.numeric(results$correct)
    stopifnot(!interactive || !is.nan(mean(correct)))
    
    if (interactive) {
      interp_scores = get_PC_interp_scores(reduced_embs)[results$sample_ids]
      attributes(interp_scores) = NULL
      
      print(c('mean correct: ', mean(correct)))
      hist(interp_scores)
      plot(interp_scores, smooth(correct))
    }
    print(c('answer key:', results$answer_key))
  }
}

# vocab_emb %>% add_abstractness_scores() %>% do_name_survey(lexicon, n_centers=1, interactive=T)
#source('./R_scripts/russ_emb_test.R')