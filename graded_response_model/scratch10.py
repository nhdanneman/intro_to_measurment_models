import numpy as np
import pyjags
import scipy
import matplotlib.pyplot as plt


# Let's assume we have some predictors, X, and a thing to predict, Y.
# A regression setup.
# Wrinkle: X also predicts var(Y) [maybe]
# So heteroskedastic regression.
# Let's do this Bayesian.
# This script will quickly simulate some example data and model it for demonstrative purposes.


# some variables related to Y and var(y)

N = 300

x1 = np.random.normal(0,1,N)
x2 = np.random.uniform(0,3,N)

error_magnitudes = 1.5 + 0.3*x1 + 0.4*x2
error_magnitudes = [max(i, 0.01) for i in error_magnitudes]

y = []
for i in range(N):
    y_i = 2 + 1.1*x1[i] + 2.1*x2[i] + np.random.normal(0,error_magnitudes[i],1)
    y.append(y_i[0])


JAGS_model = '''
  model {
    for ( i in 1:N ) {
        y[i] ~ dnorm(mu[i], tau[i])
        mu[i] <- b0m + b1m*x1[i] + b2m*x2[i]
        
        # NOTE: JAGS/BUGS uses 
        tau[i] <- 1/(sigma[i]*sigma[i])
        sigma[i] <- max(0.01, raw_sigma[i])
        raw_sigma[i] <- b0s + b1s*x1[i] + b2s*x2[i]
        
    }
    
    b0m ~ dnorm(0, .1) # note these 0.1's are PRECISION not SD
    b1m ~ dnorm(0, .1)
    b2m ~ dnorm(0, .1)
    
    b0s ~ dnorm(0, .1)
    b1s ~ dnorm(0, .1)
    b2s ~ dnorm(0, .1)

  }
  '''

model = pyjags.Model(JAGS_model, data=dict(y=y, x1=x1, x2=x2,  N=N),
                     chains=2, adapt=1500)

model.update(1500)

samples = model.sample(1000, vars=['b0m', 'b1m', 'b2m', 'b0s', 'b1s', 'b2s'])


np.median(samples['b0s']) # 0.05
np.median(samples['b1s']) # -0.07
np.median(samples['b2s']) # 0.01


### Ok, so now we have a model for the accuracy of the mapping from X -> Y.
# We can use to see where in X our predictions are bad, informing IV&V (somehow?)

# design space
x1 = np.random.normal(0,1,N)
x2 = np.random.uniform(0,3,N)

# Latin Cube or Random Search here:
x1_samples = np.random.normal(min(x1), max(x1), 100)
x2_samples = np.random.normal(min(x2), max(x2), 100)

variance_model = np.array([np.median(samples['b0s']), np.median(samples['b1s']), np.median(samples['b2s'])])

x0 = np.ones(100)

design_space = np.array([x0,x1_samples, x2_samples])

predicted_variance = np.matmul(variance_model, design_space)


