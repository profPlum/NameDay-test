if (!is.null(sys.frame(1)$ofile)) {
  .script.dir <- dirname(sys.frame(1)$ofile)
  setwd(.script.dir)
}

library(tidyverse)
data = read_csv('../logs/correlation_test.csv') %>% 
  select(-X1)

summary(data)

plot_data = data %>% mutate(names=paste0(name1, '/', name2)) %>% filter(nchar(names)<10) %>% 
  group_by(names) %>% summarise_all(mean) %>% ungroup %>% sample_n(25)

# make dist_r & name_sim bar plot
plot_data %>% select(names, dist_r, name_sim) %>% pivot_longer(-names) %>%
  ggplot(aes(x=names)) + geom_col(aes(y=value, group=names, color=name)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# make dist_r & recoded_name_sim bar plot
plot_data %>% select(names, dist_r, recoded_name_sim) %>% pivot_longer(-names) %>%
  ggplot(aes(x=names)) + geom_col(aes(y=value, group=names, color=name)) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

print(c("correlation of name_sim to dim_r:", cor(data$name_sim, data$dist_r)))
print(c("correlation of recoded_name_sim to dim_r:", cor(data$recoded_name_sim, data$dist_r)))