.script.dir <- dirname(sys.frame(1)$ofile)
setwd(.script.dir)

source('survey_api.R')
source('emb_funcs.R')
library(tidyverse)
library(tm)

reload = T

if (!exists('vocab_emb') || reload) {
  # vocab_emb_data = load_corpus('./data/Joseph-Conrad-Chance.txt', './logs/lexicon_conrad.csv', 500,
  #                          extra_stop_words=extra_stop_words, term_mat = 'tcm', use_cache=F)
  vocab_emb_data = load_vocab_emb('./logs/vocab_emb.txt', './logs/com/lexicon.csv',
                                  english_only=T, defined_only=F)
  dtm = vocab_emb_data$dtm
  lexicon = vocab_emb_data$lexicon
  vocab_emb = norm_emb(vocab_emb_data$vocab_emb)
}

#vocab_emb = simplest_forms(vocab_emb)

# b4 & after comparison with max interp scores
compare_interp_scores(vocab_emb, dtm)
interp_vocab_emb = vocab_emb %>% hard_max_emb_interp()
print(c('mean cos diff :', mean_cos_sim_diff(vocab_emb, interp_vocab_emb)))
compare_interp_scores(interp_vocab_emb, dtm)

PC_names = get_PC_names(interp_vocab_emb)
colnames(interp_vocab_emb) = rownames(PC_names)
maximal_examples = get_maximal_examples(interp_vocab_emb)

do_name_survey(interp_vocab_emb, lexicon, n_centers=1, interactive=F, use_definitions=F, n_questions = 27)