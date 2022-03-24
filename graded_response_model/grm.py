# Test out building data for and estimating a graded response model

import numpy as np
import pyjags
import scipy
import matplotlib.pyplot as plt

def data_creator(N=100, n_det=5):
    x = np.random.uniform(0, 1, N)
    b0 = np.random.uniform(-0.01, 0.01, n_det)  # 0.00
    b1 = np.random.uniform(0.5, 1.5, n_det)
    cut2 = np.random.uniform(0.99, 1.0, n_det)  # 1.00
    diffs = np.random.uniform(.3, .6, n_det)
    cut1=cut2-diffs

    mat = np.zeros((N,n_det))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            xb = b0[j] + x[i]*b1[j]
            low = scipy.stats.norm.cdf(cut1[j], xb, 1)
            med = scipy.stats.norm.cdf(cut2[j], xb, 1) - low
            hi = 1 - scipy.stats.norm.cdf(cut2[j], xb, 1)
            prs = [low, med, hi]
            yval = np.random.choice([1,2,3], 1, p=prs)
            mat[i,j] = yval

    return mat, x, b1, cut1

# create some data that is has col_mins of 1 and col_maxs of 3
# so we don't have to deal with special cases right away
N = 1000
n_det = 5

good_data = False
while good_data == False:
    mat, x, b1, cut1 = data_creator(N=N, n_det=n_det)
    colmax = np.apply_along_axis(max, 0, mat)
    colmin = np.apply_along_axis(min, 0, mat)
    if any(colmax != 3) or any(colmin != 1):
        print("making data again")
        print(colmax)
        print(colmin)
        good_data = False
    else: good_data = True


nLevs = 3

code = '''
  model {
    for ( i in 1:N ) {
        for ( j in 1:n_detectors ) {
    
          y[i,j] ~ dcat( pr[i,j,1:nLevs] )
          mu[i,j] <- 0 + b1[j]*x[i]
          
          pr[i,j,1] <- pnorm( thresh1[j] , mu[i,j] , 1 )
    
          pr[i,j,2] <- max( 0 ,  pnorm( thresh2[j] , mu[i,j] , 1 )
                               - pnorm( thresh1[j] , mu[i,j] , 1 ) )
    
          pr[i,j,3] <- 1 - max(0, pnorm( thresh2[j] , mu[i,j] , 1 ) )
    
    
          }
        }

    for (j in 1:n_detectors){
        thresh2[j] ~ dnorm(1, 3)   # basically fixed
        thresh1[j] ~ dnorm(thresh2[j]-0.5, 0.1)
        b1[j] ~ dnorm(1,.1)
    }
    
    for (i in 1:N){
      x[i] ~ dunif(0,1)
    }

  }
  '''

model = pyjags.Model(code, data=dict(y=mat, n_detectors=n_det, nLevs=nLevs, N=N),
                     chains=4, adapt=1000)


model.update(1000)


samples = model.sample(1000, vars=['thresh1', 'thresh2', 'x', 'b1'])

# some checks:
x_est = np.apply_over_axes(np.mean, samples['x'], [1,2]).flatten()
plt.scatter(x_est, x)

thresh1_est = np.apply_over_axes(np.mean, samples['thresh1'], [1,2]).flatten()

plt.scatter(thresh1_est, cut1)

b1_est = np.apply_over_axes(np.mean, samples['b1'], [1,2]).flatten()

plt.scatter(b1_est, b1)

plt.hist(samples['thresh1'][0,:,:].flatten())
plt.axvline(cut1[0], color='k', linestyle='dashed', linewidth=1)


np.median(samples['thresh1'])
np.median(samples['thresh2'])
np.median(samples['x'])



