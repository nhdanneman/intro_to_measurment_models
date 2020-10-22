### This script accompanies the "Intro to Measurement Models" Talk
### author: Nathan Danneman
### created: Oct 15, 2020
### last edited: Oct 15, 2020

# We'll use JAGS for this tutorial: http://mcmc-jags.sourceforge.net/
require(rjags)

root <- "/Users/ndanneman/Documents/personal/gits/intro_to_measurment_models"

# JAGS lets you pretty freely specify a complex Bayesian model
# It figures out the sampling distributions, and generates samples


################################## 
### Example 1: Bayesian means

# Suppose we have some iid measurements. 
# What is the mean and sd of the distribution from which this data was drawn?
x <- c(3, 5, 9, 9, 12, 12)
hist(x)

# The sample mean is the maximum likelihood estimator
ml_mean <- mean(x)
abline(v=ml_mean, col="red", lwd=3)

# What if you think this data was drawn from a distribution with a higher mean?
# But you aren't TOO sure?
# Suppose your prior is normally distributed, mean=10, sd=3

# This is a JAGS model, we write it to file.
cat(
  "
    # opens with the 'model' and open-brace
    model{
    # observations drawn from normal distribution with mean and precision
    for (i in 1:N){
      x[i] ~ dnorm(mu, tau)
    }
    # prior on the mean
    mu ~ dnorm(10, .5) 
    # deterministic link (arrow, not tildea)
    tau <- pow(sigma,-2)
    # prior on sigma
    sigma ~ dnorm(5, 5)
  }", file=paste0(root, "/code/models/mean.txt")
)

N <- length(x)
nchains=3
# We pass the model file, the data, the number of chains to run, and a short pre-burn-in to jags.model
jags <- jags.model(file=paste0(root, "/code/models/mean.txt"),
                   data=list(x=x, N=N),
                   n.chains=nchains, n.adapt=100)

# This is a burn-in period
# Let's the sampler get out of weird starting conditions
update(jags, 2000)

# This is MCMC sampling
# We tell jags.samples which parameters to record into output
iter <- 1000
out <- jags.samples(jags,
                    c('mu', 'tau', 'sigma'),
                    1000)

# out is a list, holding info on the parameters we asked it to track
is.list(out)  # TRUE
names(out) # tracks the parameters we asked
# we can obtain mean values simply:
dim(out$mu) # ARRAY of shape: parameters X iterations X chains
# but if you just call it, you get the average
out$mu

# ALL values in 'out' are legit samples from posteriors
#   and thus we can use them to describe paramteres
# Here are all the estimates of mu, in a matrix of ITERATIONS x CHAINS shape
mu_est <- matrix(out$mu, nrow=iter, ncol=nchains)

hist(mu_est, main="Samples from Estimated Posterior Distribution")

# MCMC samplers can fail in a couple ways:
# get "stuck" in areas of low probability
# models that are under-identified in terms of scale or rotation
# Line graph, per chain, helps us tell if we got "good mixing" (~~ convergence check)
plot(mu_est[,1], type="l")
points(mu_est[,2], type="l", col="red")
points(mu_est[,3], type="l", col="blue")

# Check out the coda package to get some canned diagnostics

# So, did we get a reasonable estimate?
hist(x)
abline(v=ml_mean, col="red", lwd=3)
abline(v=mean(mu_est), col="blue", lwd=3)
legend("topleft", col=c("red", "blue"), legend = c("ML", "Bayesian"), lty=c(1,1), lwd=3)


#############################################
### RE: Priors
# So, priors, how do you choose them? Where do they come from? Do they matter?
# Note: Frequentist methods have an implied prior -- there's no getting around "picking one"
# They can be extremely vague/uninformative/flat.
# They can encode SME expertise.
# They can come from other models/estimates.

# The more data/evidence you have (i.e. the stronger the likelihood), the less they matter.
# If they really move your inference, you likely have weak data.

# Let's re-specify our Bayesian means model with some different priors (same data):
cat(
  "model{
  for (i in 1:N){
  x[i] ~ dnorm(mu, tau)
  }
  mu ~ dnorm(10, .1) 
  tau <- pow(sigma,-2)
  sigma ~ dnorm(5, 5)
  }", file=paste0(root, "/code/models/mean_diffuse.txt")
)

N <- length(x)
nchains=3
# TODO: describe, comment this
jags <- jags.model(file=paste0(root, "/code/models/mean_diffuse.txt"),
                   data=list(x=x, N=N),
                   n.chains=nchains, n.adapt=100)
# TODO: what differntiaties this following from previous "adapt"
update(jags, 2000)

# TODO: what differentiates "samples" from "update" from "n.adapt"??
iter <- 1000
mdiffuse <- jags.samples(jags,
                    c('mu', 'tau', 'sigma'),
                    1000)

cat(
  "model{
  for (i in 1:N){
  x[i] ~ dnorm(mu, tau)
  }
  mu ~ dnorm(2, .5) 
  tau <- pow(sigma,-2)
  sigma ~ dnorm(5, 5)
  }", file=paste0(root, "/code/models/mean_low.txt")
)

N <- length(x)
nchains=3
# TODO: describe, comment this
jags <- jags.model(file=paste0(root, "/code/models/mean_low.txt"),
                   data=list(x=x, N=N),
                   n.chains=nchains, n.adapt=100)
# TODO: what differntiaties this following from previous "adapt"
update(jags, 2000)

# TODO: what differentiates "samples" from "update" from "n.adapt"??
iter <- 1000
mlow <- jags.samples(jags,
                        c('mu', 'tau', 'sigma'),
                        1000)

# Some footwork to visualize the data, various priors and mean(posteriors)
hist(x, xlim=c(-3, 15), freq=FALSE)
# prior = N(10, 4)
prior_dat <- rnorm(1000, 10, 4)
pd <- density(prior_dat)
lines(pd$x, pd$y, col="magenta", lwd=2)
prior_dif <- rnorm(1000, 10, 1/.1^2)
pd_dif <- density(prior_dif)
lines(pd_dif$x, pd_dif$y, col="orange", lwd=2)
prior_low <- rnorm(1000, 2, 1/.5^2)
pd_low <- density(prior_low)
lines(pd_low$x, pd_low$y, col="blue", lwd=2)

abline(v=out$mu[1], lwd=2, col="magenta")
abline(v=mdiffuse$mu[1], lwd=2, col="orange")
abline(v=mlow$mu[1], lwd=2, col="blue")
abline(v=mean(x), lwd=3, col="red", lty=2)
legend("topleft", col=c("magenta", "orange", "blue", "red"), 
       legend = c("Reasonable", "Diffuse", "Low", "ML"), lty=c(1,1, 1, 2), lwd=3)


##############################################################
### Quick Chat: Why do Bayesian stuff, again?

### OK, so we've learned how to estimate a mean in a Bayesian setting. Cool.
### The reasons to go Bayesian are: 
  # flexibility (we'll see that in a bit)
  # inferential rigor
  # small data and meaningful priors

# Let's look at Bayesian OLS quickly to set the stage for measurement models
cat(
  "model{
  for (i in 1:N){
    # this is the likelihood function, Y is distributed normally, conditioned on a mean (with an sd)
    y[i] ~ dnorm(mu[i], tau)
    # the mean is a linear function of an intercept (b0) and a slope (b1)
    mu[i] <- b0 + b1*x
    # x is data -- no likelihood model or prior
    # y is modeled as a probabilistic outcome, thus needs a likelihood model
  }
  # priors on b0, b1, and the sd term
  b0 ~ dnorm(0, .1)
  b1 ~ dnorm(0, .1)
  tau <- pow(sigma,-2)
  sigma ~ dnorm(5, 5)
  }", file=paste0(root, "/code/models/OLS.txt")
)


#################################################
### Measurement Models

# In a measurement model, we presume a latent variable exists.
# We also presume it causes some observables

# We'll work the following example for concreteness:
# Classical test theory: your score on a test is the fraction of questions you got right
# Item Response Theory: your latent aptitude determines if you get questions right.
# Each question might be differently difficult and discriminating

# Let's simulate some data like this:
n_questions <- 10
n_students <- 10
latent_aptitude <- rnorm(n_students, 0, 1)
per_question_difficulty <- rnorm(n_questions, 0, 1.5)
per_question_discrimination <- runif(n_questions, .4, 1.2)

# Probability a student got a question right is f(ability_s, diff_q, discr_q)
outcomes <- matrix(0, nrow=n_students, ncol=n_questions)
for (i in 1:n_students){
  for (j in 1:n_questions){
    xb <- latent_aptitude[i] * per_question_discrimination[j] + per_question_difficulty[j]
    pr <- exp(xb) / (1+exp(xb))
    outcomes[i,j] <- rbinom(1, 1, pr)
  }}

outcomes

# Let's write the model:
cat(
  "model{
    for (i in 1:n_students){
      for (j in 1:n_questions){
        
        # we think each cell in our matrix is distributed bernoulli
        outcomes[i,j] ~ dbern(pr[i,j])
        
        # the probability is the logit of...
        # ... student-specific aptitude (row fixed effects) ...
        # ... and question-specific discr and diff (col fixed effects)
        logit(pr[i,j]) <- aptitude[i] * discr[j] + diff[j]
        # note: logit == exp(x) / (1+exp(x))
      }
    }
    # let's use a for-loop to add priors on aptitude 
    for (i in 1:n_students){
      aptitude[i] ~ dnorm(0, 1)
    }
    # similar trick for discr and diff
    for (j in 1:n_questions){
      discr[j] ~ dnorm(1,1) # note, discr should be positive!!
      diff[j] ~ dnorm(0,1)
    }
  
  }", file=paste0(root, "/code/models/IRT.txt")
)


## Ok, let's estimate that thing. Similar to before.
nchains=3
# We pass the model file, the data, the number of chains to run, and a short pre-burn-in to jags.model
jags <- jags.model(file=paste0(root, "/code/models/IRT.txt"),
                   data=list(outcomes=outcomes, n_students=n_students, n_questions=n_questions),
                   n.chains=nchains, n.adapt=100)

# This is a burn-in period
# Let's the sampler get out of weird starting conditions
update(jags, 2000)

# This is MCMC sampling
# We tell jags.samples which parameters to record into output
iter <- 1000
mod <- jags.samples(jags,
                    c('aptitude', 'diff', 'discr'),
                    1000)

# reminder: outputs are parameterShape X iterations X chains
dim(mod$aptitude)  # N students X 1000 iterations X 3 chains

# Latent aptitude should correlate with our specified latent aptitude
plot(latent_aptitude, apply(mod$aptitude, 1, median),
     xlab="True Aptitude", ylab="Estimated Aptitude")
# These might not look great. But note in this toy example we have 
# 2*p + N parameters to estimate from N*p binary data points.
 # More data in the N direction *really* helps.

# Let's build more intuition here about the theoretical model:
low <- min(mod$aptitude)
hi <- max(mod$aptitude)
possible_apt <- seq(low, hi, 0.05)

est_diff <- apply(mod$diff, 1, median)
est_discr <- apply(mod$discr, 1, median)

xb <- possible_apt*est_discr[1] + est_diff[1]
pr <- exp(xb) / (1+exp(xb))
plot(possible_apt, pr, type="l", lwd=2, ylim=c(0,1),
     xlab = "Plausible Range of Aptitudes",
     ylab = "pr(correct)")
rug(apply(mod$aptitude, 1, median))

for (i in 2:min(n_questions, 10)){
  xb <- possible_apt*est_discr[i] + est_diff[i]
  pr <- exp(xb) / (1+exp(xb))
  points(possible_apt, pr, type="l", lwd=2, col=i)
}
  
## Ok, so how different are our fancy measurement model predictions than simpler ones?
## Simpler = more or different assumptions? Or both?
## assume all questions have same diff and perfectly discr: classical test theory
## guess at the relative difficulty of questions: questions with different "point values" 
##   (i.e. an index)
## try to infer everything: measurement model
simple_average_score <- apply(outcomes, 1, mean)
modeled_aptitude <- apply(mod$aptitude, 1, median)
plot(simple_average_score, modeled_aptitude,
     xlab="Simple Average Scoring", ylab="Modeled Aptitude")
## Better question: In what cases might these be different?



# TODO: Example or discussion of identificaiton. Scale and rotational invariance.



