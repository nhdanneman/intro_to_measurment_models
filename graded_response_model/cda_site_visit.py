import numpy as np
import pandas as pd
import pyjags
import pickle

if __name__ == "__main__":

    print("Prep and read data")
    dat = pd.read_csv('/Users/ndanneman/Documents/dataMachines/CHASE/irt_historical/siem_dump.csv')

    # make the matrix of (distinct observables) by (detectors)

    unique_detectors = dat['performers'].unique()
    n_detectors = len(unique_detectors)

    unique_observables = dat['observables'].unique()
    n_observables = len(unique_observables)

    mat = np.zeros((n_observables,n_detectors))

    # fill the matrix with multinomial levels
    for i in range(dat.shape[0]):
        p = dat['performers'][i]
        lev = dat['levels'][i]
        obs = dat['observables'][i]
        mat_col = np.where(unique_detectors == p)[0][0]
        mat_row = np.where(unique_observables == obs)[0][0]
        mat[mat_row,mat_col] = lev

    mat = mat + 1


    # infer max from data (NOTE can't do this in prod!!)
    colmax = np.apply_along_axis(max, 0, mat)


    code = '''
      model {
        for ( i in 1:N ) {
            for ( j in 1:n_detectors ) {
    
              y[i,j] ~ dcat( pr[i,j,1:colmax[j]] )
              mu[i,j] <- 0 + b1[j]*x[i]
    
              # first level
              pr[i,j,1] <- pnorm( thresh[j,1] , mu[i,j] , 9.1 ) # note tau 9.1 ~~ sigma 0.33
    
              # second through penultimate levels
              for (k in 2:(colmax[j]-1)){
                pr[i,j,k] <- max( 0 ,  pnorm( thresh[j,k] , mu[i,j] , 9.1 )
                                   - pnorm( thresh[j,(k-1)] , mu[i,j] , 9.1 ) )
                }
    
              # top/last level
              pr[i,j,colmax[j]] <- 1 - max(0, pnorm( thresh[j,(colmax[j]-1)] , mu[i,j] , 9.1 ) )
    
    
              }
            }
    
    
        for (j in 1:n_detectors){
            thresh[j,1] ~ dnorm(0,.1)
            for (k in 2:(colmax[j]-1)) {
              thresh[j,k] ~ dnorm(thresh[j, (k-1)] + 0.2, 0.1)
              }
            }
    
        for (j in 1:n_detectors){
            b1[j] ~ dnorm(1,.1)
        }
    
        for (i in 1:N){
          x[i] ~ dunif(0,1)
        }
    
      }
      '''

    # colmax[colmax==1] = 2

    print("Prep model")

    model = pyjags.Model(code, data=dict(y=mat, n_detectors=mat.shape[1], colmax=colmax, N=mat.shape[0]),
                         chains=2, adapt=10)


    model.update(10)

    print("Estimate model")

    samples = model.sample(10, vars=['thresh', 'x', 'b1'])


    # need to output two things: model params and entities
    # to get scores, just agg over x:
    x_est = np.apply_over_axes(np.mean, samples['x'], [1, 2]).flatten()

    print("Output scoring")

    dets = []
    for i in unique_observables:
        this = ''
        hits = list(set(list(dat['performers'][dat['observables']==i])))
        for j in hits:
            this = this + ", " + j
        dets.append(this)


    d = {'Score': x_est, "Entity": unique_observables, "Detections":dets}
    entity_output = pd.DataFrame(d)
    entity_output = entity_output.sort_values('Score')

    entity_output.to_csv("/Users/ndanneman/Documents/dataMachines/CHASE/irt_historical/model_results.csv", index=False)


    with open("/Users/ndanneman/Documents/dataMachines/CHASE/irt_historical/model_file.p", "wb") as output_file:
        pickle.dump(samples, output_file)

    print("Mission Accomplished!")

