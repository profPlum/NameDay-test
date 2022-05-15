install.packages('tidyverse', dependencies = T,
                 repos='https://cran.rstudio.com/')
library(tidyverse)

survey_data = read_csv('./data/Word Set Naming Task.csv')
survey_data = survey_data[-1,]

answer_key = c("chagrin", "ascended", "unforeseen", "morgue", "jerome", "bliss",
               "unearned", "grads", "reissue", "bosque", "commences", "entomology",
               "flared", "argent", "daft", "horsepower",  "insurer", "raj", "yeh", 
               "esplanade", "blacksmith", "dacia", "packer", "bushels", "couriers",
               "keyword", "lander")

q_ids = grep('\\d+ . ', colnames(survey_data))
names(survey_data) = gsub('\\d+ . please name this set: ', '', names(survey_data))
names(answer_key) = colnames(survey_data)[q_ids]

long_data = survey_data %>% select(names(answer_key)) %>% pivot_longer(everything()) %>% rename(answer=value)
answer_dist = long_data %>% group_by(name, answer) %>% summarise(n=n()) %>% ungroup %>%
  mutate(correct=answer==answer_key[name]) %>% group_by(name, correct) %>%
  mutate(rank=sort(n,decreasing=T,index.return=T)$ix) %>% ungroup %>% mutate(rank=as.integer(if_else(correct, 1, rank+1))) %>% 
  rename(rank_correct_first=rank)

# tie is when top answer is a toss up between correct answer & random answer
answer_dist = answer_dist %>% group_by(name) %>% arrange(name, desc(n)) %>% mutate(rank=1:5) %>% mutate(tie=sum(n[correct]==n)>1) %>% ungroup

answer_dist %>% filter(!tie) %>% filter(correct) %>% group_by(rank) %>% summarise(n=sum(n)) %>% ungroup %>% 
mutate(percent=n/sum(n)*100) %>% ggplot(aes(x=rank,y=percent)) + geom_col() + ggtitle('Rank Distributions of Correct Answers')
# pie(no_ties$n, labels=no_ties$rank, main='Rank Distributions of Correct Answers')

# most_freq_answers = answer_dist %>% group_by(name) %>% filter(rank==1 & !tie)
# %>% arrange(name, desc(correct), desc(n))

stopifnot(length(unique(answer_dist$name))*5==length(answer_dist$name))

answer_dist %>% filter(!tie) %>% mutate(percent=(n/sum(n))*100) %>% ggplot() +
  geom_col(aes(x=rank, y=percent, fill=correct)) + ggtitle('Rank Distributions of All Answers')

# answer_dist %>%  mutate(percent=(n/sum(n))*100) %>% mutate(percent=(n/sum(n))*100) %>% ggplot() +
#   geom_col(aes(x=rank_correct_first, y=percent, fill=correct)) +
#   ggtitle('Survey Answer Distribution Plot')

# sorted by accuracy for top & bottom info
answer_dist = answer_dist %>% group_by(name) %>% mutate(accuracy=sum(n*correct)/sum(n)) %>% arrange(accuracy) %>% 
  mutate(name_short=as_vector(map(strsplit(name, ','), ~paste0(paste(.x[1:3], collapse=','), '...'))))

question_samples = sample(unique(answer_dist$name), 9)#tail(unique(answer_dist$name),2)
answer_dist %>% group_by(name) %>% ungroup %>% filter(name %in% question_samples) %>%
  ggplot() + facet_wrap(~name_short, scale='free_x') + geom_col(aes(x=answer, y=n, color=correct)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + ylab('Counts') + ggtitle('Sample Question Answer Distributions')

