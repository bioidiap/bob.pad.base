
'''Runs error analysis on score sets, outputs metrics and plots'''

import bob.measure.script.figure as measure_figure
import bob.bio.base.script.figure as bio_figure
from .error_utils import calc_threshold

ALL_CRITERIA = ('bpcer20', 'eer', 'min-hter')


class Metrics(bio_figure.Metrics):
    '''Compute metrics from score files'''

    def __init__(self, ctx, scores, evaluation, func_load,
                names=('FtA', 'APCER', 'BPCER', 'FAR', 'FRR', 'HTER')):
        super(Metrics, self).__init__(
            ctx, scores, evaluation, func_load, names
        )

    def get_thres(self, criterion, dev_neg, dev_pos, far):
        if self._criterion == 'bpcer20':
            return calc_threshold('bpcer20', dev_neg, dev_pos)
        else:
            return super(Metrics, self).get_thres(
                criterion, dev_neg, dev_pos, far)


class MultiMetrics(measure_figure.MultiMetrics):
    '''Compute metrics from score files'''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(MultiMetrics, self).__init__(
            ctx, scores, evaluation, func_load,
            names=('FtA', 'APCER', 'BPCER', 'FAR', 'FRR', 'ACER'))

    def get_thres(self, criterion, dev_neg, dev_pos, far):
        if self._criterion == 'bpcer20':
            return calc_threshold('bpcer20', dev_neg, dev_pos)
        else:
            return super(MultiMetrics, self).get_thres(
                criterion, dev_neg, dev_pos, far)


class Roc(bio_figure.Roc):
    '''ROC for PAD'''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Roc, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = ctx.meta.get('x_label') or 'APCER'
        default_y_label = '1-BPCER' if self._semilogx \
            else 'BPCER'
        self._y_label = ctx.meta.get('y_label') or default_y_label


class Det(bio_figure.Det):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Det, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = ctx.meta.get('x_label') or 'APCER (%)'
        self._y_label = ctx.meta.get('y_label') or 'BPCER (%)'


class Hist(measure_figure.Hist):
    ''' Histograms for PAD '''

    def _setup_hist(self, neg, pos):
        self._title_base = 'PAD'
        self._density_hist(
            pos[0], n=0, label='Bona-fide', color='C1'
        )
        self._density_hist(
            neg[0], n=1, label='Presentation attack', alpha=0.4, color='C7',
            hatch='\\\\'
        )
