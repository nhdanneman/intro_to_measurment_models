'''
This should be a well-documented, clean version of the CDA model estimation code.



Inputs:
 - a longform detection-time list (see example format below)
 - a config file showing how to read the detections (see examples)

Outputs:
 - model parameters that can be used by the inference file/container
 - scored entity-time tuples

'''

import numpy as np
import pandas as pd
import pyjags
import pickle
import json
import scipy.stats as stats


# helper function that takes in a config dict, detector and score, and emits an integer "level"
# some cludgy assumptions here:
# in the config, continuous things are floats, categorical things, even if numbers, as strings
def detector_score_to_int(conf, det, score):
    lev = -1
    # handle the case of categorical levels
    if conf[det]['ctype'] == "categorical":
        # levels should be one-based, so add 1
        lev = conf[det]['info'].index(score) + 1
        return lev
    # handle the case of continuous levels
    else:
        score = float(score)
        for i in range(len(conf[det]['info'])):
            if score <= conf[det]['info'][i]:
                lev = i + 1
                return lev
        if lev == -1:
            lev = len(conf[det]['info'])
            return lev
    print("something went wrong")
    return 0


if __name__ == "__main__":


    print("Read the siem_data")
    # Assumes the historical detection data has the following fields and info...
    # detector: the name of the detector. Must match the config file.
    # score: the score from the detector. Might be a string, int, or float.
    # timestamp: a string in yyyy-mm-dd format
    # observable: the entity that was detected, e.g. "1.2.3.4" or "john_doe_user" or "badstuff.com"
    historical_detections = pd.read_csv('/data/inputs/siem_dump.csv')



    print("Read the config file")
    # Detectors typically output continuous or categorical scores
    # e.g. 0-100
    # e.g. "low", "medium", "critical"
    # the config file is a mapping, per detector, from the detector space into an ordinal space
    #   because the Graded Response Model requires ordinal, integer input
    # each ROW of the config has
    ## detector name (must match the data file!)
    ## detector output type (one of "continuous" or "categorical"
    ## mappings
    #### if "categorical": list of strings in increasing severity or confidence
    ####   e.g. ['low', 'medium', 'severe']
    ####   this would map to [1, 2, 3] with 0 as "I saw nothing here"
    #### if "continuous": list of cutpoints, should be no more than 3 or 4 realistically, to cut the
    ####   continuous space into ordinal groupings
    ####   e.g. [0.3, 0.7, 0.9]
    ####   this would map 0.01-0.3 as 1, 0.31-0.7 as 2, 0.71-0.9 as 3 and > 0.9 as 4

    with open('/data/inputs/config.json') as f:
        config = json.load(f)


    # The input the the model is a matrix of (distinct) observables by detectors.
    # Infer the number of detectors from the number of keys in teh config
    unique_detectors = list(config.keys())
    n_detectors = len(unique_detectors)

    # infer the number of observables from the SIEM data dump
    unique_observables = historical_detections['observable'].unique()
    n_observables = len(unique_observables)

    # initialize the empty matrix
    mat = np.zeros((n_observables, n_detectors))


    # munge the conf into a more convenient form:
    # dict of detector -> dict with keys type -> "continuous/categorical" and info, a list of categories or cutpoints
    conf = dict()
    for i in config.keys():
        ctype = config[i][0]
        info = config[i][1:]
        conf[i] = dict()
        conf[i]['ctype'] = ctype
        conf[i]['info'] = info


    # Iterate over the SIEM data (detector, score, timestamp, observable)
    # Populate the relevant matrix entry (observable, detector) with an int representing the detector level

    for i in range(historical_detections.shape[0]):
        detector = historical_detections['detector'][i]
        lev = detector_score_to_int(conf, detector, historical_detections['score'][i])
        obs = historical_detections['observable'][i]
        mat_col = unique_detectors.index(detector)
        mat_row = np.where(unique_observables == obs)[0][0]
        # handle case where this observable-detector cell already observed
        mat[mat_row,mat_col] = max(mat[mat_row,mat_col], lev)

    # The formulation of the GRM I wrote needs the lowest level to be 1
    # but the code above has 0 as "this detector did not fire at all against this observable"
    # so here we fix that
    mat = mat + 1


    # The algo needs to know the max possible value for each detector
    # Use the config to get that info, and keep it in same order
    # This is the max in matrix, which has 1 added value.
    # That is, in the matrix, 1 means "no report", that's why we add 2 (1) to continuous (categorical) vals
    colmax = []
    for i in conf.keys():
        if conf[i]['ctype']=="continuous":
            top = len(conf[i]['info']) + 2
        else:
            top = len(conf[i]['info']) + 1
        colmax.append(top)


    # A generalized GRM model
    # dynamic number of detectors, number of observations, and number of levels per detector
    # this version is identified by the following:
    #   - fix sigma, the sd of the normal distribution in the ordered probit models, to 0.33
    #   - fix the intercept at zero
    #   - allow x to range between zero and one
    #   - put ascending priors on successive thresholds, but no hard constraints
    #   - allow the slope term to be freely estimated, but with positive and moderately tight prior

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


    print("Preparing model")

    model = pyjags.Model(code, data=dict(y=mat, n_detectors=mat.shape[1], colmax=colmax, N=mat.shape[0]),
                         chains=2, adapt=13)

    print("Continuing to burn in model")
    model.update(12)

    print("Estimating model")

    samples = model.sample(11, vars=['thresh', 'x', 'b1'])


    #### Wrap-up and outputs.
    # We need to output two things:
    # 1) a csv of observables and their scores
    #   - this supports SOCs running this once, or CPT teams using it ad hoc
    # 2) the model parameters, which can be read into the inference-only script
    #   - this supports SOC users who want a daily check

    # to get per-observable scores, just aggregate over chains and samples:
    # note, _samples_ is of shape [length of thing, number of samples, number of chains]
    x_est = np.apply_over_axes(np.median, samples['x'], [1, 2]).flatten()

    # also push out which detectors hit on this observable, for convenience
    # a more thorough version would also include what level/score was for each detector
    dets = []
    for i in unique_observables:
        this = ''
        hits = list(set(list(historical_detections['detector'][historical_detections['observable']==i])))
        for j in hits:
            this = this + ", " + j
        dets.append(this)

    d = {'Score': x_est, "Entity": unique_observables, "Detections":dets}
    entity_output = pd.DataFrame(d)
    entity_output = entity_output.sort_values('Score')

    # This is a dataframe containing each observable/entity, it's score, and what detectors fired on it.
    # This is the core output for one-off uses of CDA
    entity_output.to_csv("/data/inputs/model_results.csv", index=False)


    print("Writing out model paramters")
    # Here's the lazy way to write out all the sample info
    # It might be useful for iterative updating, or as priors moving forward
    with open("/data/inputs/model_file.p", "wb") as output_file:
        pickle.dump(samples, output_file)


    # Build a good, parsable data structure to store model parameters.
    # For every detector in the config, we need the b1 and thresholds.
    #   ( b0 and sigma are fixed, and x is assumed to lie in [0,1] )
    # Build a dict that we'll turn into a JSON object, with each detector a key.
    # Detectors will have a b1 key and thresh key, the latter is a list.
    # Nice because each detector can have different numbers of thresholds

    model_dict = dict()

    # Iterate over detectors, collecting up their b1 and threshold values into nested dict
    for idx,val in enumerate(conf.keys()):
        det = val
        model_dict[det] = dict()
        model_dict[det]['thresh'] = []
        model_dict[det]['b1'] = -10.0
        # each detector has colmax-1 number of thresholds
        for j in range(colmax[idx]-1):
            threshold_samples = samples['thresh'][idx, j, :, :].flatten()
            thresh_median = np.median(threshold_samples)
            model_dict[det]['thresh'].append(thresh_median)
        b1_samples = samples['b1'][idx,:,:].flatten()
        b1_median = np.median(b1_samples)
        model_dict[det]['b1']=b1_median

    # Dump the model dict as a json string to file.
    json_object = json.dumps(model_dict, indent = 4)
    with open('/data/inputs/model_dict.json', 'w') as outfile:
        outfile.write(json_object)

    # To support inference, we likely want to pre-calculated per-detector likelihood matrices
    # This makes inference fast; don't need to recompute over and over. (memoisation)
    # Data structure is just a dict of matrices, keyed on detector.
    # Each matrix has rows for possible levels of an analytic, and colums for 0.01 to 0.99 by 0.01.
    # Each element is log probability of observing level given x == column value.
    det_log_prs_dict = dict()
    for i in model_dict.keys():
        b1 = model_dict[i]['b1']
        thresholds = model_dict[i]['thresh']
        possible_xvals = np.arange(0.01, 0.99, .01)
        mat = np.zeros((len(thresholds)+1, len(possible_xvals)))
        for j in range(len(possible_xvals)):
            for k in range(len(thresholds)+1):
                # bottom category
                if k == 0:
                    m = possible_xvals[j]*b1
                    pr = stats.norm.cdf(thresholds[k], m, .33)
                # middle categories
                if (k>0) & (k < len(thresholds)):
                    m = possible_xvals[j] * b1
                    top_pr = stats.norm.cdf(thresholds[k], m, .33)
                    bot_pr = stats.norm.cdf(thresholds[k-1], m, .33)
                    pr = top_pr - bot_pr
                # top category
                if k == len(thresholds):
                    pr = 1 - stats.norm.cdf(thresholds[k-1], m, .33)
                # log the probability and add it to the matrix
                pr = max(pr, 0.00000001)
                logpr = np.log(pr)
                mat[k,j] = logpr
        det_log_prs_dict[i] = mat


    print("Writing out likelihood parameters for fast inference")
    with open("/data/inputs/inference_matrices.p", "wb") as output_file:
        pickle.dump(det_log_prs_dict, output_file)

    # Always give yourself a pat on the back if your code ran!
    print("Mission Accomplished!")


