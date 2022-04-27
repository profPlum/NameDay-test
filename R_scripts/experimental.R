greedy_PC_name_selection = function(vocab_emb) {
  # we set the first member to be the vocab vector with the highest
  # PC dimension of all (and thereby most aligned with its axis)
  target_PC_names = apply(vocab_emb, -2, max) %>% which.max
  text_names = rownames(vocab_emb)[target_PC_names]
  target_PC_names = as.matrix(vocab_emb[target_PC_names,])

  for (i in 2:dim(vocab_emb)[2]) {
    sim = vocab_emb %*% target_PC_names
    ortho_loss = apply(sim, -2, norm, type='2') # two norm prioritizes minizization of max similarity
    next_best = which.min(ortho_loss)
    target_PC_names = cbind(target_PC_names, vocab_emb[next_best,])
    text_names = c(text_names, rownames(vocab_emb)[next_best])
  }
  colnames(target_PC_names) = text_names
  return(t(target_PC_names)) # we want standard row format
}

# NOTE: this is essentially computing Russian interp score!
PC_name_orthog_loss = function(V) {
  PC_names = t(V)%*%V
  eye = diag(dim(PC_names)[1])
  eye_approx = t(PC_names)%*%PC_names
  print(mean(diag(eye_approx))/mean(eye_approx))
  image.real(eye_approx)
  loss =-sum(diag(eye_approx))#norm(abs(eye_approx-eye))
  print(c('log(det(I)): ', log(det(eye_approx))))
  return(loss)
}

set_names = function(vocab_emb) {
  Q = t(solve(PC_names %*% solve(greedy)))
  vocab_emb = vocab_emb %*% Q
  colnames(vocab_emb) = rownames(greedy)
  PC_names = t(vocab_emb)%*%vocab_emb
  stopifnot(all.equal(PC_names, greedy%*%Q))
  return(norm_emb(vocab_emb))
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
  cos_sim_plot = function(cos_sims_V, cos_sims_O, n_plot) {
    cos_sims_plot_V = cos_sims_V[1:n_plot, 1:n_plot]
    cos_sims_plot_O = cos_sims_O[1:n_plot, 1:n_plot]
    coors_V = cmdscale(-cos_sims_plot_V-min(-cos_sims_plot_V))
    coors_O = cmdscale(-cos_sims_plot_O-min(-cos_sims_plot_O))
    axis_lim = 0.4
    
    coors = cbind(coors_V, coors_O)
    p = ggplot(as_tibble(coors), aes(x=V1, y=V2)) + geom_segment(aes(xend=V3, yend=V4, color='yellow', alpha=0.3),
                                                                 arrow = arrow(length = unit(0.5, "cm"))) +
      geom_text(aes(x=(V1+V3)/2,y=(V2+V4)/2,label=rownames(coors)), position = 'dodge', check_overlap = T) + 
      theme(legend.position = "none") + xlim(-axis_lim, axis_lim) + ylim(-axis_lim, axis_lim)
    print(p)
  }
  
  ids = sample(dim(V)[1], sample_sz)
  O = norm_emb(O[ids,])
  V = V[ids,]
  
  n_plot = 10
  cos_sims_V = V%*%t(V)
  cos_sims_O = O%*%t(O)
  cos_sim_plot(V, O, n_plot)
  
  cos_sim_differences = cos_sims_O - cos_sims_V
  hist(cos_sim_differences)
  return(mean(abs(cos_sim_differences)))
}


V = vocab_emb
O = hard_max_emb_interp(vocab_emb)
PC_name_orthog_loss(O)
PC_name_orthog_loss(V)
cos_sim_diff(V, O)

compare_interp_scores(V)
compare_interp_scores(O)

View(get_maximal_examples(O))

givens = function(i,j, theta, n) {
  G = diag(n)
  G[i,i] = G[j,j] = cos(theta)
  G[i,j] = -sin(theta)
  G[j,i] = -G[i,j]
  return(G)
}

cos_sim = function(vec, vec2) vec%*%vec2/(norm(vec,type='2')*norm(vec2, type='2'))

do_rotation_test = function(p1=pi/8, p2=pi/8, plane_dims=1:4) {
  vec = runif(5)
  G = givens(plane_dims[1],plane_dims[2], p1, 5)
  vec1 = G%*%vec
  G2 = givens(plane_dims[3],plane_dims[4], p2, 5)
  vec2 = G2%*%vec1
  return(acos(cos_sim(vec, vec2)))
}



