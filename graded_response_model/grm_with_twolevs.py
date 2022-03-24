'''
Question: can we arbitrarily add a third level to binary data
 so that we can use this code as written?

The weird part is that the second threshold will have nothing to
 ground it to reality...


'''

# Test out building data for and estimating a graded response model

import numpy as np
import pyjags
import scipy
import matplotlib.pyplot as plt


def data_creator(N=100, n_levs=[2, 3, 4]):
    n_det = len(n_levs)
    x = np.random.uniform(0, 1, N)
    b0 = np.random.uniform(-0.01, 0.01, n_det)  # 0.00
    b1 = np.random.uniform(.9, 1.5, n_det)
    # cuts should be a list of lists
    # each list will have ascending cutpoints
    cuts = []
    for j in range(n_det):
        j_cuts = []
        jcut_diff = 1.0 / (n_levs[j] - 1)
        j_cuts.append(1.0)
        for k in range(n_levs[j] - 2):
            next_cut_center = j_cuts[0] - jcut_diff
            next_cut = np.random.uniform(next_cut_center - .1, next_cut_center + .1)
            j_cuts.insert(0, next_cut)
        cuts.append(j_cuts)
    mat = np.zeros((N, n_det))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            xb = b0[j] + x[i] * b1[j]
            if len(cuts[j]) == 1:
                low = scipy.stats.norm.cdf(cuts[j][0], xb, .33)
                hi = 1 - scipy.stats.norm.cdf(cuts[j][0], xb, .33)
                prs = [low, hi]
                yval = np.random.choice([1, 2], 1, p=prs)
                mat[i, j] = yval
            else:
                prs = []
                for k in range(len(cuts[j]) + 1):
                    # pr(k == 1)
                    if len(prs) == 0:
                        next_pr = scipy.stats.norm.cdf(cuts[j][0], xb, .33)
                        prs.append(next_pr)
                        continue
                    # pr(k == highest)
                    if len(prs) == len(cuts[j]):
                        next_pr = 1 - scipy.stats.norm.cdf(cuts[j][k - 1], xb, .33)
                        prs.append(next_pr)
                    # intermediate prs
                    else:
                        top = scipy.stats.norm.cdf(cuts[j][k], xb, .33)
                        bot = scipy.stats.norm.cdf(cuts[j][k - 1], xb, .33)
                        next_pr = top - bot
                        prs.append(next_pr)
                possible_vals = list(range(1, len(prs) + 1))
                yval = np.random.choice(possible_vals, 1, p=prs)
                mat[i, j] = yval

    return mat, x, b1, cuts


# create some data that is has col_mins of 1 and col_maxs of 3
# so we don't have to deal with special cases right away
N = 500
n_levs = [2, 2, 2, 3, 4]

good_data = False
while good_data == False:
    mat, x, b1, cut1 = data_creator(N=N, n_levs=n_levs)
    colmax = np.apply_along_axis(max, 0, mat)
    colmin = np.apply_along_axis(min, 0, mat)
    if any(colmax > 5) or any(colmin != 1):
        print("making data again")
        print(colmax)
        print(colmin)
        good_data = False
    else:
        good_data = True

# infer max from data (NOTE can't do this in prod!!)
colmax = np.apply_along_axis(max, 0, mat)

colmax[colmax==2] = 3

code = '''
  model {
    for ( i in 1:N ) {
        for ( j in 1:n_detectors ) {

          y[i,j] ~ dcat( pr[i,j,1:colmax[j]] )
          mu[i,j] <- 0 + b1[j]*x[i]

          # first level
          pr[i,j,1] <- pnorm( thresh[j,1] , mu[i,j] , 9.1 ) # note tau 9.1 ~~ sigma 0.33

          # second through penultimate leves
          for (k in 2:(colmax[j]-1)){
            pr[i,j,k] <- max( 0 ,  pnorm( thresh[j,k] , mu[i,j] , 9.1 )
                               - pnorm( thresh[j,(k-1)] , mu[i,j] , 9.1 ) )
            }

          # top/last level
          pr[i,j,colmax[j]] <- 1 - max(0, pnorm( thresh[j,(colmax[j]-1)] , mu[i,j] , 9.1 ) )


          }
        }


    for (j in 1:n_detectors){
        thresh[j,1] ~ dnorm(0,.2)
        for (k in 2:(colmax[j]-1)) {
          thresh[j,k] ~ dnorm(thresh[j, (k-1)] + 0.2, 0.2)
          }
        }

    for (j in 1:n_detectors){
        b1[j] ~ dnorm(1,.2)
    }

    for (i in 1:N){
      x[i] ~ dunif(0,1)
    }

  }
  '''

model = pyjags.Model(code, data=dict(y=mat, n_detectors=len(n_levs), colmax=colmax, N=N),
                     chains=4, adapt=1000)

model.update(1000)

samples = model.sample(1000, vars=['thresh', 'x', 'b1'])

# some checks:
x_est = np.apply_over_axes(np.mean, samples['x'], [1, 2]).flatten()
plt.scatter(x_est, x)

# where does it pust thresh...?
samples['thresh'].shape

np.mean(samples['thresh'][0, 0, :, :])
np.mean(samples['thresh'][0, 1, :, :])
np.mean(samples['thresh'][0, 2, :, :])
np.mean(samples['thresh'][1, 0, :, :])
np.mean(samples['thresh'][1, 1, :, :])
np.mean(samples['thresh'][1, 2, :, :])
np.mean(samples['thresh'][4, 0, :, :])
np.mean(samples['thresh'][4, 1, :, :])
np.mean(samples['thresh'][4, 2, :, :])

np.mean(samples['b1'][4])
plt.scatter(range(len(samples['b1'][4,:,0])),samples['b1'][4,:,0])
plt.scatter(range(len(samples['b1'][4,:,0])),samples['b1'][4,:,1])
plt.scatter(range(len(samples['b1'][4,:,0])),samples['b1'][4,:,2])
plt.scatter(range(len(samples['b1'][4,:,0])),samples['b1'][4,:,3])
plt.axhline(y=b1[4], linewidth=4, color='b')
plt.axhline(y=np.mean(samples['b1'][4,:,:]), linewidth=4, color='g')
