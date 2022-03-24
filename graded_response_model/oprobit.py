import numpy as np
import pyjags
import scipy
import matplotlib.pyplot as plt

### make data supporting a single o-probit

N = 100
x = np.random.uniform(0, 1, N)

intercept = 0
beta = 1.9

cut1 = 1

cut2 = 2

xb = intercept + x*beta

y = []
for i in xb:
    low = scipy.stats.norm.cdf(cut1, i, 1)
    med = scipy.stats.norm.cdf(cut2, i, 1) - low
    hi = 1 - scipy.stats.norm.cdf(cut2, i, 1)
    prs = [low, med, hi]
    yval = np.random.choice([0,1,2], 1, p=prs)
    y.append(yval[0])

y = np.array(y) + 1

# plt.scatter(x,xb)
# plt.scatter(xb,y)

nLevs = 3


code = '''
  model {
    for ( i in 1:N ) {
      y[i] ~ dcat( pr[i,1:nLevs] )
      mu[i] <- 0 + 1.9*x[i]
      pr[i,1] <- pnorm( thresh1 , mu[i] , 1 )
      
      pr[i,2] <- max( 0 ,  pnorm( thresh2 , mu[i] , 1 )
                           - pnorm( thresh1 , mu[i] , 1 ) )
                           
      pr[i,3] <- 1 - max(0, pnorm( thresh2 , mu[i] , 1 ) )
                                             
        
      }
    
    thresh1 ~ dnorm(1, .1)
    thresh2 ~ dnorm(thresh1 + 0.5, .1)
    
  }
  '''


model = pyjags.Model(code, data=dict(y=y,x=x, N=N,
                                     nLevs=nLevs),
                     chains=4, adapt=1000)

model.update(1000)

samples = model.sample(1000, vars=['thresh1', 'thresh2'])

np.median(samples['thresh1'])
np.median(samples['thresh2'])







### now, let b0 and b1 float
### see if things still identified by x in (0,1) and sigma = 1

code = '''
  model {
    for ( i in 1:N ) {
      y[i] ~ dcat( pr[i,1:nLevs] )
      mu[i] <- 0 +  b1*x[i]
      pr[i,1] <- pnorm( thresh1 , mu[i] , 1 )

      pr[i,2] <- max( 0 ,  pnorm( thresh2 , mu[i] , 1 )
                           - pnorm( thresh1 , mu[i] , 1 ) )

      pr[i,3] <- 1 - max(0, pnorm( thresh2 , mu[i] , 1 ) )


      }

    thresh1 ~ dnorm(1, .1)
    thresh2 <- 2
    b1 ~ dnorm(0,1)

  }
  '''

model = pyjags.Model(code, data=dict(y=y, x=x, N=N,
                                     nLevs=nLevs),
                     chains=4, adapt=1000)

model.update(1000)

samples = model.sample(1000, vars=['thresh1', 'thresh2',  'b1'])

np.median(samples['thresh1'])
np.median(samples['thresh2'])
np.median(samples['b1'])

plt.scatter(list(range(len(samples['b0'].flatten()))), samples['b0'].flatten())