# plot_spc.R

# Susceptibility density and largest cluster sizes against temperature, as in
# Figs. 2, 5, 7 of Blatt, Wiseman & Domany (1997). `out` has one row per
# temperature and point, with columns temp, susceptibility, point and label.

plot_susceptibility <- function(out) {
  n <- length(unique(out$point))
  by_temp <- split(out, out$temp)
  temp <- sapply(by_temp, function(d) d$temp[1])
  chi_density <- sapply(by_temp, function(d) d$susceptibility[1] * d$temp[1] / n)
  plot(temp, chi_density, type = "b", pch = 20,
       xlab = "T", ylab = expression(chi * T / N))
  invisible(out)
}

plot_clusters <- function(out, n_clusters = 4) {
  n <- length(unique(out$point))
  by_temp <- split(out, out$temp)
  temp <- sapply(by_temp, function(d) d$temp[1])
  sizes <- t(sapply(by_temp, function(d)
    sort(as.vector(table(d$label)), decreasing = TRUE)[1:n_clusters]))

  matplot(temp, sizes, type = "b", pch = 20, lty = 1, col = 1:n_clusters, ylim = c(0, n),
          xlab = "T", ylab = "cluster size")
}
