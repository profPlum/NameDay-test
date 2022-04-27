get_PC_names_max = function(vocab_emb) {
  name_idx = apply(vocab_emb, -1, which.max)
  PC_names = vocab_emb[name_idx,]
  return(PC_names)
}

PC_names = get_PC_names(vocab_emb)
PC_names_max = get_PC_names_max(vocab_emb)

forward_err = NULL
max_err = NULL
diff = NULL
for (i in 1:100) {
  vocab_emb_noisy = sweep(vocab_emb, 2, rnorm(dim(vocab_emb)[2])) %>% norm_emb()
  PC_names1 = get_PC_names(vocab_emb_noisy)
  PC_names_max1 = get_PC_names_max(vocab_emb_noisy)
  forward_err = c(forward_err, norm(PC_names-PC_names1))
  max_err = c(max_err, norm(PC_names_max-PC_names_max1))
  diff = c(diff, norm(PC_names_max1-PC_names1))
}

hist(forward_err)
boxplot(forward_err, main='forward_err')
hist(max_err)
boxplot(max_err, main='max_err')
hist(diff)
boxplot(diff)

maximize = function(A) {
  A_norms = apply(A, -1, norm, type='2')
  ans = solve(A,diag(A_norms))
  return(ans)
}
