#rm(list=ls())
setwd('~/PycharmProjects/wordEmbeddings')
debugSource('./R_scripts/survey.R')
debugSource('./R_scripts/emb_funcs.R')
library(tidyverse)
library(tm)

# person_names = read_csv('./data/baby_names/NationalNames.csv') %>% arrange(desc(Count)) %>% head(2000)
# person_names = unique(person_names$Name)
# save(person_names, file='./logs/person_names.RData')
#load('./logs/person_names.RData')
#extra_stop_words = c(person_names, 'go', 'that','anyhow', 'however', 'also', 'almost', 'agre', 'fyne', 'elizabeth', 'roderick')

reload = T

if (!exists('vocab_emb') || reload) {
  # vocab_emb = load_corpus('./data/Joseph-Conrad-Chance.txt', './logs/lexicon_conrad.csv', 500,
  #                          extra_stop_words=extra_stop_words, term_mat = 'tcm', use_cache=F)
  vocab_emb_data = load_vocab_emb('./logs/com/vocab_emb_com.txt', './logs/com/lexicon_com.csv',
                                  english_only=T, defined_only=F)
  dtm = vocab_emb_data$dtm
  lexicon = vocab_emb_data$lexicon
  vocab_emb = norm_emb(vocab_emb_data$vocab_emb)
}

#vocab_emb = simplest_forms(vocab_emb)

# b4 & after comparison with max interp scores
#compare_interp_scores(vocab_emb, dtm)
interp_vocab_emb = vocab_emb %>% hard_max_emb_interp()
print(c('mean cos diff :', mean_cos_sim_diff(vocab_emb, interp_vocab_emb)))
print(c('orthog loss interp: ', PC_name_orthog_loss(interp_vocab_emb)))
print(c('orthog loss vanilla: ', PC_name_orthog_loss(vocab_emb)))

compare_interp_scores(interp_vocab_emb, dtm)

PC_names = get_PC_names(interp_vocab_emb)
colnames(interp_vocab_emb) = rownames(PC_names)
maximal_examples = get_maximal_examples(interp_vocab_emb)

do_name_survey(interp_vocab_emb, lexicon, n_centers=1, interactive=F, use_definitions=F, n_questions = 27)

# #:param max_portion_med_dist: is max portion of the median distance that is not considered an outlier
# #:param min_coors: is the minimum number of coordinates to preserve while iterating
# #:param iterate: whether to iterate
# trim_outliers = function(coors, min_coors=nrow(coors)/3, iterate=T, max_portion_med_dist=1.5) {
#   find_outliers = function(coors, max_portion_med_dist=1.5) {
#     center = apply(coors, -1, median)
#     print(c('center:',center))
#     coor_distance = abs(coors-center)
#     coor_distance_cutoff = apply(coor_distance, -1, median)*max_portion_med_dist
#     print(c('coor distance cutoof:', coor_distance_cutoff))
#     outliers = coor_distance > coor_distance_cutoff
#     outliers = apply(outliers,-2, any) # verified to work
#     return(outliers)
#   }
#   
#   # prune outliers until we have just enough data points
#   new_coors = coors
#   while (nrow(coors)>min_coors) {
#     coors = new_coors
#     print(c('remaining points:',length(coors)))
#     outlier_mask = find_outliers(coors, max_portion_med_dist)
#     if (!any(outlier_mask)) break
#     new_coors = coors[!outlier_mask,]
#     if (!iterate) { coors = new_coors; break }
#     # if we don't want to iterate
#   }
#   return(coors)
# }
# 
# plot_PCs = function(coors) {
#   stopifnot(dim(coors)[2]==2)
#   coors = trim_outliers(coors, iterate=T)
#   
#   (clusters = kmeans(coors, centers=6, nstart=25))
#   
#   # from here: https://uc-r.github.io/kmeans_clustering
#   library(factoextra)
#   print(fviz_cluster(clusters, geom = "point", 
#                      data = as_tibble(coors)) +
#           geom_text(aes(label=rownames(coors)), position = 'dodge', check_overlap = T))
# }

# TODO: test this with 2 PCs
# plot_PCs(abs(interp_vocab_emb))

