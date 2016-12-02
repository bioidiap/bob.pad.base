'''
Created on 16 Nov 2016

@author: sbhatta
'''

import sys, os
import pkg_resources
import bob.io.base
import numpy as np
import bob.measure
from bob.pad.base.evaluation import PadIsoMetrics


#def main(arguments):
def main(command_line_parameters=None):

    scorefile_devel = pkg_resources.resource_filename('bob.pad.base', 'test/data/pad_devel_replaymobile_IqmScores_SVM.hdf5')
    scorefile_test  = pkg_resources.resource_filename('bob.pad.base', 'test/data/pad_test_replaymobile_IqmScores_SVM.hdf5')

#     PAI_labels = [('mattescreen-photo', 'A'), ('mattescreen-video', 'A'), ('print-fixed', 'A'), ('print-hand','A') ]

    rms = PadIsoMetrics.PadIsoMetrics() # PadIsoMetrics(PAI_labels)

    devel_dict = rms.load_scores_hdf5(scorefile_devel)
    test_dict  = rms.load_scores_hdf5(scorefile_test)
    
    threshEER_dev, dev_far, dev_frr = rms.eer(devel_dict)
    
    eer_devel = 50.0*(dev_far + dev_frr)
    print('threshEER_dev (grandtest): %s'  % threshEER_dev)
    print('FRR, FAR (devel): %s %s'  % (dev_frr, dev_far))
    print('EER (%%): %.3f%%'  % eer_devel)
    
    test_hter, test_far, test_frr = rms.hter(test_dict, threshEER_dev)
    print "     * FAR : %.3f%%" % (100*test_far)
    print "     * FRR : %.3f%%" % (100*test_frr)
    print "     * HTER: %.3f%%" % (100*test_hter)
    
    test_bpcer = 100.0*rms.bpcer(test_dict, threshEER_dev)
    print('BPCER from dict: %.3f%%'  % test_bpcer )
    
    bf_scores = test_dict['real'][0]
    test_bpcer = 100.0*rms.bpcer(bf_scores, threshEER_dev)
    print('BPCER from np-array: %.3f%%'  % test_bpcer )
    
    attack_apcer, attack_perf_dict =  rms.apcer(test_dict, 'C', threshEER_dev)
    print('\nAPCER: %.3f%%'  % (100.0*attack_apcer) )
    print('Performance for individual PAIs:')
    for k in attack_perf_dict.keys():
        print('%s: %.3f%%' %(k, 100.0*attack_perf_dict[k]))


'''
'''
if __name__ == '__main__':
    main(sys.argv[1:])
