### This script accompanies the "Intro to Measurement Models" Talk
### author: Nathan Danneman
### created: Oct 15, 2020
### last edited: Oct 15, 2020

# We'll use JAGS for this tutorial http://mcmc-jags.sourceforge.net/
require(rjags)

root <- "/Users/ndanneman/Documents/personal/gits/intro_to_measurment_models"

# JAGS lets you pretty freely specify a complex Bayesian model
# It figures out the sampling distributions, and generates samples

## Example 1: Bayesian means

# Suppose we have some iid measurements. 
# What is the mean and sd of the distribution from which this data was drawn?
x <- c(3, 5, 9, 9, 12)
hist(x)

# The sample mean is the maximum likelihood estimator
ml_mean <- mean(x)
abline(v=ml_mean, col="red", lwd=3)

# What if you think this data was drawn from a distribution with a higher mean?
# But you aren't TOO sure?
# Suppose your prior is normally distributed, mean=10, sd=3

cat(
  "model{
    for (i in 1:N){
      x[i] ~ dnorm(mu, tau)
    }
    mu ~ dnorm(10, .01) 
    tau <- pow(sigma,-2)
    sigma ~ dnorm(5, 5)
  }", file=paste0(root, "/code/models/mean.txt")
)

N <- length(x)
nchains=3
jags <- jags.model(file=paste0(root, "/code/models/mean.txt"),
                   data=list(x=x, N=N),
                   n.chains=nchains, n.adapt=100)
update(jags, 2000)

iter <- 1000

out <- jags.samples(jags,
                    c('mu', 'tau', 'sigma'),
                    1000)

# out is a list, holding info on the parameters we asked it to track
is.list(out)
names(out)

mu_est <- matrix(out$mu, nrow=iter, ncol=nchains)
plot(mu_est[,1], type="l")
points(mu_est[,2], type="l", col="red")
points(mu_est[,3], type="l", col="blue")

hist(x)
abline(v=ml_mean, col="red", lwd=2)
abline(v=mean(mu_est), col="blue", lwd=2)


