'''
Created on 16 Nov 2016

@author: sbhatta
'''

import sys, os
import bob.io.base
import numpy as np
import bob.measure

class PadIsoMetrics():

    def __init__(self):
        """ constructor. """

        self.real_name = 'bonafide' #real_presentation_name #'real'
        self.attack_name = 'attack' #attack_presentation_name #'attack'
        
    def save_scores_hdf5(self, outfile, scores_dict):
        """ saves input scores_dict dictionary in a hdf5 formatted file"""

        h5out = bob.io.base.HDF5File(outfile, "w")
    
        for p in scores_dict.keys():
            if len(scores_dict[p]) == 1: # real_set
                h5out.set(p, scores_dict[p][0] )
                h5out.set_attribute('presentation', self.real_name, p)
            else:
                #write attacks
                h5out.set(p, scores_dict[p][0] )
                h5out.set_attribute('attack_potential', scores_dict[p][1], p)
                h5out.set_attribute('presentation', self.attack_name, p)
        
        del h5out

    def load_scores_hdf5(self, infile):
        """ loads a hdf5 file, and trys to construct a dictionary of scores. Returns the score-dictionary."""

        h5in = bob.io.base.HDF5File(infile, "r")
         
        scores_dict = {}         
        h5in.cd('/')
        class_labels = h5in.keys(relative='True')
        for p in class_labels:
            scores = h5in.get(p)
            attrs = h5in.get_attributes(p)
            if len(attrs) == 2: #then the two elements are 'presentation' and 'attack_potential'
                ap = attrs['attack_potential']
                scores_dict[p] = [scores, ap] 
            else:
                scores_dict[p] = [scores]
            
        del h5in
        return scores_dict  
        
    
    def eer(self, scores_dict):
        """ computes EER threshold using the scores in the supplied dictionary 
        Input:
        scores_dict: dictionary where each key is the name of the presentation ('real' or one attack-type), 
        and the corresponding value is a tuple: (scores, attack_potential).
        'scores' should be a 1D numpy-array of floats containing scores
        'attack_potential' should be one of the 3 letters 'A', 'B', or 'C')
        Scores for 'real' presentations will not have an associated 'attack_potential',
        so, if the value of a key is a tuple of length 1, the key-value pair is assumed
        to represent a 'real'-presentation set.
        Return:
        tuple of three floats: (eer_threshold, far, frr). These are computed using functions from bob.measure.
        """

        real_scores = None
        attack_scores = None
        assert scores_dict is not None, 'no development score-set provided for computing EER'    
        
        for k in scores_dict.keys():
            keyvalue = scores_dict[k]
            if len(keyvalue)==2:
                if attack_scores is None:
                    attack_scores = scores_dict[k][0]
                else:
                    attack_scores = np.concatenate((attack_scores, scores_dict[k][0]))
            else:
                if len(keyvalue)==1:
                    real_scores = scores_dict[k][0]
        
        assert (attack_scores is not None), 'Empty attack-scores list. Cannot compute EER'
        assert (real_scores is not None), 'Empty real-scores list. Cannot compute EER.'
        self.threshEER_dev = bob.measure.eer_threshold(attack_scores, real_scores)
        
        self.dev_far, self.dev_frr = bob.measure.farfrr(attack_scores, real_scores, self.threshEER_dev)
#         self.eer_devel = 50.0*(self.dev_far + self.dev_frr)
#         print('eer()::threshEER: %s' % self.threshEER_dev)
        return (self.threshEER_dev, self.dev_far, self.dev_frr)


    def hter(self, scores_dict, score_threshold):
        """ computes HTER on test-set scores, using the supplied score-threshold.
        Inputs: 
        scores_dict: dictionary where each key is the name of the presentation ('real' or one attack-type), 
        and the corresponding value is a tuple: (scores, attack_potential).
        'scores' should be a 1D numpy-array of floats containing scores
        'attack_potential' should be one of the 3 letters 'A', 'B', or 'C')
        Scores for 'real' presentations will not have an associated 'attack_potential',
        so, if the value of a key is a tuple of length 1, the key-value pair is assumed
        to represent a 'real'-presentation set.
        score_threshold: (float) value to be used for thresholding scores.
        Return:
        tuple of three floats: (hter, far, frr). These are computed using functions from bob.measure.
        """

        assert ((score_threshold is not None) and isinstance(score_threshold, (int, long, float)) ), 'input score_threshold should be a number (float or integer).'
        
        real_scores = None
        attack_scores = None
        assert scores_dict is not None, 'no test score-set available for computing HTER'    
        
        for k in scores_dict.keys():
            key_value = scores_dict[k]
            if len(key_value)==2:
                if attack_scores is None:
                    attack_scores = scores_dict[k][0]
                else:
                    attack_scores = np.concatenate((attack_scores, scores_dict[k][0]))
            else:
                if len(key_value)==1:
                    real_scores = scores_dict[k][0]
       
        assert (attack_scores is not None), 'Empty attack-scores list. Cannot compute EER'
        assert (real_scores is not None), 'Empty real-scores list. Cannot compute EER.'
        test_far, test_frr = bob.measure.farfrr(attack_scores, real_scores, score_threshold)
#         test_good_neg = bob.measure.correctly_classified_negatives(attack_scores, score_threshold).sum()
#         test_good_pos = bob.measure.correctly_classified_positives(real_scores, score_threshold).sum()
        hter = (test_far+test_frr)/2.0
        
        return (hter, test_far, test_frr)


    def _check_attack_potential(self, attack_potential):
        """ For now, we assume three levels of attack-potential: 'C'>'B'>'A' """

        if attack_potential is None:
            attack_potential = 'C'
        if attack_potential not in ['A', 'B', 'C']:
            attack_potential = 'C'
        
        return attack_potential
    

    def bpcer(self, scores, score_threshold=0.0):
        """ computes BPCER  on test-set scores, using either the supplied score-threshold, 
        or the threshold computed from the EER of the development set 
        Inputs:
        scores: a 1D numpy-array of scores corresponding to genuine (bona-fide) presentations.
        score_threshold: a floating point number specifying the score-threshold to be used for deciding accept/reject.
        
        Return:
        floating-point number representing the bpcer computed for the input score-set
        """

        bonafide_scores = None
        if isinstance(scores, dict):
            #extract 'real' scores from dictionary
            for k in scores.keys():
                key_value = scores[k]
                if len(key_value) == 1:
                    bonafide_scores = key_value[0]
        else:
            #verify that scores is a 1D numpy array
            if isinstance(scores, np.ndarray) and len(scores.shape)==1:
                bonafide_scores = scores
        
        assert bonafide_scores is not None, 'input scores does not contain bona-fide scores, for computing BPCER.'
        assert isinstance(score_threshold, (int, long, float)), 'input score_threshold should be a number (float or integer).'
        
        correct_scores = bonafide_scores[bonafide_scores<score_threshold].shape[0]
        
        return correct_scores/float(bonafide_scores.shape[0]) 
    

    def apcer(self, scores_dict, attack_potential='C', score_threshold=0.0):
        """computes APCER as defined in ISO standard. For now, we assume three levels of attack-potential: 'C'>'B'>'A' 
        
        Inputs:
        scores_dict: a dictionary where each key corresponds to a specific PAI (presentation-attack-instrument)
        Keys corresponding to PAIs will have as value a list of 2 elements: 
        1st element: a 1D numpy-array of scores
        2nd element: a single letter 'A', 'B', or 'C', specifying the attack-potential of the PAI.
                             
        attack_potential: a letter 'A', 'B', or 'C', specifying the attack_potential at which the APCER is to be computed
        score_threshold: a floating point number specifying the score-threshold to be used for deciding accept/reject.
             
        Returns:
        tuple consisting of 2 elements:
        1st element: apcer at specified attack-potential
        2nd element: dictionary of hter of individual PAIs that have attack-potential at or below input-parameter attack_potential.
        """
        
        attack_potential = self._check_attack_potential( attack_potential)

        attack_perf_dict = {} #dictionary to store the hter for each attack-type that is at or below specified attack-potential
        result_list = []
        for k in scores_dict.keys():
            if len(scores_dict[k]) == 2: #consider only the keys where the value is a list of 2 elements
                if scores_dict[k][1] <= attack_potential:
                    scores =  scores_dict[k][0]
                    result = (scores[scores>=score_threshold].shape[0])/float(scores.shape[0])
                    result_list.append(result)
                    attack_perf_dict[k]=result
        
        return (max(result_list), attack_perf_dict)  

