source('R_scripts/emb_funcs.R')
library(tidyverse)

# Not currently used...
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
    cat(question_id, '. ')
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
                          n_centers=1, n_questions=10, interactive=T, use_definitions=T) {
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
      PCA = prcomp(vocab_emb_cluster, rank=rank)
      reduced_embs = get_vocab_emb(PCA, rank, term_mat = 'emb')
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


#################### Make Survey ####################

reload = T

if (!exists('vocab_emb') || reload) {
  # vocab_emb_data = load_corpus('../data/Joseph-Conrad-Chance.txt', '../logs/lexicon_conrad.csv', 500,
  #                          extra_stop_words=extra_stop_words, term_mat = 'tcm', use_cache=F)
  vocab_emb_data = load_vocab_emb('./logs/vocab_emb.txt', './logs/lexicon.csv',
                                  english_only=T, defined_only=F)
  dtm = vocab_emb_data$dtm
  lexicon = vocab_emb_data$lexicon
  vocab_emb = norm_emb(vocab_emb_data$vocab_emb)
}

#vocab_emb = simplest_forms(vocab_emb)

# b4 & after comparison with max interp scores
compare_interp_scores(vocab_emb, dtm)
interp_vocab_emb = vocab_emb %>% hard_max_emb_interp()
print(c('mean cos diff :', cos_sim_diff(vocab_emb, interp_vocab_emb)))
compare_interp_scores(interp_vocab_emb, dtm)

PC_names = get_PC_names(interp_vocab_emb)
colnames(interp_vocab_emb) = rownames(PC_names)
maximal_examples = get_maximal_examples(interp_vocab_emb)

do_name_survey(interp_vocab_emb, lexicon, n_centers=1, interactive=F, use_definitions=T, n_questions = 27)

##################################################