rm(list=ls())
library(tidyverse)

source("R/other_functions.R")
source("R/sampler.R")
Rcpp::sourceCpp("src/sampler.cpp")

# Generating data
n_ <- 1000
set.seed(42)
# Simulation 1
x <- matrix(seq(-pi,pi,length.out = n_))
x_new <- matrix(seq(-pi,pi,length.out = n_))
colnames(x) <- "x"
colnames(x_new) <- "x"
y <- sin(3*x) + rnorm(n = n_,sd = 0.1)

y <- y[x>0,,drop = FALSE]
x <- x[x>0,,drop = FALSE]

sp_mod <- rsp_sampler(x_train = x,y = y,nIknots = 100,df = 3,
                      sigquant = 0.9,delta = 1,nu = 2,
                      a_delta = 0.0001,d_delta = 0.0001,
                      n_mcmc = 2500,n_burn = 500,
                      scale_y = TRUE)

# Formatting the sampler plot
par(mfrow=c(1,1))
plot(x,y,main = "P-Splines robust priors")
quantiles_y_hat <- apply(sp_mod$y_train_post,2,function(x){quantile(x,probs = c(0.025,0.5,0.975))})
lines(x,sin(3*x), col = "red")
lines(x,quantiles_y_hat[2,],col = "blue")
lines(x,quantiles_y_hat[1,],lty = "dashed", col = "blue")
lines(x,quantiles_y_hat[3,],lty = "dashed", col = "blue")

# Traceplots
par(mfrow=c(2,2))
plot(sp_mod$beta_0_post,type = "l", main = expression(beta[0]))
plot(sp_mod$tau_b_post,type = "l", main = expression(tau[b]))
plot(sp_mod$delta_post,type = "l", main = expression(delta))
plot(sp_mod$tau_post,type = "l", main = expression(tau))

