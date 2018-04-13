'''Runs error analysis on score sets, outputs metrics and plots'''

import pkg_resources  # to make sure bob gets imported properly
import logging
import click
import numpy as np
import  bob.measure.script.figure as measure_figure
from tabulate import tabulate
import matplotlib.pyplot as mpl
from bob.extension.scripts.click_helper import verbosity_option
from  bob.measure.utils import (get_fta, get_thres)
from bob.measure import (
    far_threshold, eer_threshold, min_hter_threshold, farfrr
)
from . import error_utils

ALL_CRITERIA = ('bpcer20', 'eer', 'min-hter')

def calc_threshold(method, neg, pos):
    """Calculates the threshold based on the given method.
    The scores should be sorted!

    Parameters
    ----------
    method : str
        One of ``bpcer201``, ``eer``, ``min-hter``.
    neg : array_like
        The negative scores. They should be sorted!
    pos : array_like
        The positive scores. They should be sorted!

    Returns
    -------
    float
        The calculated threshold.

    Raises
    ------
    ValueError
        If method is unknown.
    """
    method = method.lower()
    if method == 'bpcer20':
        threshold = far_threshold(neg, pos, 0.05, True)
    elif method == 'eer':
        threshold = eer_threshold(neg, pos, True)
    elif method == 'min-hter':
        threshold = min_hter_threshold(neg, pos, True)
    else:
        raise ValueError("Unknown threshold criteria: {}".format(method))

    return threshold

class Metrics(measure_figure.Metrics):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Metrics, self).__init__(ctx, scores, evaluation, func_load)

    ''' Compute metrics from score files'''
    def compute(self, idx, dev_score, dev_file=None,
                eval_score=None, eval_file=None):
        ''' Compute metrics for the given criteria'''
        dev_neg, dev_pos, _, eval_neg, eval_pos, _ =\
                self._process_scores(dev_score, eval_score)

        title = self._titles[idx] if self._titles is not None else None
        headers = ['' or title, 'Development %s' % dev_file]
        if self._eval and eval_score is not None:
            headers.append('Eval. % s' % eval_file)
        for m in ALL_CRITERIA:
            raws = []
            threshold = calc_threshold(m, dev_neg, dev_pos)
            click.echo("\nThreshold of %f selected with the %s criteria" % (
                threshold, m))
            apcer, bpcer = farfrr(dev_neg, dev_pos, threshold)
            raws.append(['BPCER20', '{:>5.1f}%'.format(apcer * 100)])
            raws.append(['EER', '{:>5.1f}%'.format(bpcer * 100)])
            raws.append(['min-HTER', '{:>5.1f}%'.format((apcer + bpcer) * 50)])
            if self._eval and eval_neg is not None:
                apcer, bpcer = farfrr(eval_neg, eval_pos, threshold)
                raws[0].append('{:>5.1f}%'.format(apcer * 100))
                raws[1].append('{:>5.1f}%'.format(bpcer * 100))
                raws[2].append('{:>5.1f}%'.format((apcer + bpcer) * 50))

            click.echo(
                tabulate(raws, headers, self._tablefmt),
                file=self.log_file
            )

class HistPad(measure_figure.Hist):
    ''' Histograms for PAD '''

    def _setup_hist(self, neg, pos):
        self._title_base = 'PAD'
        self._density_hist(
            pos, label='Bona Fide', color='C1', **self._kwargs
        )
        self._density_hist(
            neg, label='Presentation attack', alpha=0.4, color='C7',
            hatch='\\\\', **self._kwargs
        )

def _calc_pass_rate(threshold, scores):
    return (scores >= threshold).mean()

def _iapmr_dot(threshold, iapmr, real_data, **kwargs):
    # plot a dot on threshold versus IAPMR line and show IAPMR as a number
    axlim = mpl.axis()
    mpl.plot(threshold, 100. * iapmr, 'o', color='C3', **kwargs)
    if not real_data:
        mpl.annotate(
            'IAPMR at\noperating point',
            xy=(threshold, 100. * iapmr),
            xycoords='data',
            xytext=(0.85, 0.6),
            textcoords='axes fraction',
            color='black',
            size='large',
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            horizontalalignment='center',
            verticalalignment='top',
        )
    else:
        mpl.text(threshold + (threshold - axlim[0]) / 12, 100. * iapmr,
                 '%.1f%%' % (100. * iapmr,), color='C3')

def _iapmr_line_plot(scores, n_points=100, **kwargs):
    axlim = mpl.axis()
    step = (axlim[1] - axlim[0]) / float(n_points)
    thres = [(k * step) + axlim[0] for k in range(2, n_points - 1)]
    mix_prob_y = []
    for k in thres:
        mix_prob_y.append(100. * _calc_pass_rate(k, scores))

    mpl.plot(thres, mix_prob_y, label='IAPMR', color='C3', **kwargs)


def _iapmr_plot(scores, threshold, iapmr, real_data, **kwargs):
    _iapmr_dot(threshold, iapmr, real_data, **kwargs)
    _iapmr_line_plot(scores, n_points=100, **kwargs)


class HistVuln(measure_figure.Hist):
    ''' Histograms for vulnerability '''
    def _get_neg_pos_thres(self, idx, dev_score, eval_score):
        assert len(dev_score) == self._min_arg
        dev_neg_list = []
        eval_neg_list = []
        dev_pos_list = []
        eval_pos_list = []
        for i in range(self._min_arg):
            dev_neg, dev_pos, _, eval_neg, eval_pos, _ = self._process_scores(
                dev_score[i], eval_score[i]
            )
            dev_neg_list.append(dev_neg)
            dev_pos_list.append(dev_pos)
            eval_neg_list.append(eval_neg)
            eval_pos_list.append(eval_pos)

        threshold = get_thres(
            self._criter, dev_neg_list[0], dev_pos_list[0]
        ) if self._thres is None else self._thres[idx]
        return (dev_neg_list, dev_pos_list,
                eval_neg_list, eval_pos_list, threshold)

    def _setup_hist(self, neg, pos):
        self._title_base = 'Vulnerability'
        assert len(neg) == len(pos) == self._min_arg
        self._density_hist(
            pos[0], label='Bona Fide', color='C1', **self._kwargs
        )
        self._density_hist(
            neg[0], label='Zero-effort impostors', alpha=0.8, color='C0',
            **self._kwargs
        )
        self._density_hist(
            neg[1], label='Presentation attack', alpha=0.4, color='C7',
            hatch='\\\\', **self._kwargs
        )

    def _lines(self, threshold, neg, pos, **kwargs):
        if 'iapmr_line' not in self._ctx.meta or self._ctx.meta['iapmr_line']:
            #plot vertical line
            super(HistVuln, self)._lines(threshold, neg, pos)

            #plot iapmr_line
            iapmr, _ = farfrr(neg[1], pos[0], threshold)
            ax2 = mpl.twinx()
            # we never want grid lines on axis 2
            ax2.grid(False)
            real_data = True if 'real_data' not in self._ctx.meta else \
                    self._ctx.meta['real_data']
            far, frr = farfrr(neg[0], pos[0], threshold)
            _iapmr_plot(neg[1], threshold, iapmr, real_data=real_data)
            click.echo(
                'HTER (t=%.2g) = %.2f%%; IAPMR = %.2f%%' % (
                    threshold,
                    50*(far+frr), 100*iapmr
                )
            )

            ax2.set_ylabel("IAPMR (%)", color='C3')
            ax2.tick_params(axis='y', colors='red')
            ax2.yaxis.label.set_color('red')
            ax2.spines['right'].set_color('red')

class PadPlot(measure_figure.PlotBase):
    '''Base class for PAD plots'''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(PadPlot, self).__init__(ctx, scores, evaluation, func_load)
        if 'figsize' in ctx.meta:
            mpl.figure(figsize=ctx.meta['figsize'])

    def _process_scores(self, dev_score, eval_score):
        '''Process score files and return neg/pos/fta for eval and dev'''
        assert len(dev_score) == self._min_arg
        dev_neg_list = []
        eval_neg_list = []
        dev_pos_list = []
        eval_pos_list = []
        for i in range(self._min_arg):
            dev_neg, dev_pos, _, eval_neg, eval_pos, _ = \
                    super(PadPlot, self)._process_scores(
                        dev_score[i], eval_score[i]
                    )
            dev_neg_list.append(dev_neg)
            dev_pos_list.append(dev_pos)
            eval_neg_list.append(eval_neg)
            eval_pos_list.append(eval_pos)
        return (dev_neg_list, dev_pos_list, None,
                eval_neg_list, eval_pos_list, None)

    def end_process(self):
        '''Close pdf '''
        #do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
           ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    def _plot_legends(self):
        #legends for all axes
        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            li, la = ax.get_legend_handles_labels()
            lines += li
            labels += la
        mpl.gcf().legend(lines, labels, fancybox=True, framealpha=0.5)

class Epc(PadPlot):
    ''' Handles the plotting of EPC '''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Epc, self).__init__(ctx, scores, evaluation, func_load)
        if 'eval_scores_0' not in self._ctx.meta:
            raise click.UsageError("EPC requires dev and eval score files")
        self._iapmr = True if 'iapmr' not in self._ctx.meta else \
                self._ctx.meta['iapmr']
        self._title = 'EPC and IAPMR' if self._iapmr else 'EPC'
        self._x_label = r"Weight $\beta$"
        self._y_label = "WER (%)"
        self._eval = True #always eval data with EPC
        self._split = False
        self._nb_figs = 1

    def compute(self, idx, dev_score, dev_file, eval_score, eval_file=None):
        ''' Plot EPC for PAD'''
        licit_dev_neg = dev_score[0][0]
        licit_dev_pos = dev_score[0][1]
        licit_eval_neg = eval_score[0][0]
        licit_eval_pos = eval_score[0][1]
        spoof_eval_neg = eval_score[1][0]
        mpl.gcf().clear()
        epc_baseline = error_utils.epc(
            licit_dev_neg, licit_dev_pos, licit_eval_neg,
            licit_eval_pos, 100
        )
        mpl.plot(
            epc_baseline[:, 0], [100. * k for k in epc_baseline[:, 1]],
            color='C0',
            label=self._label(
                'WER', '%s-%s' % (dev_file[0], eval_file[0]), idx
            ),
            linestyle='-'
        )
        mpl.xlabel(self._x_label)
        mpl.ylabel(self._y_label)
        if self._iapmr:
            mix_prob_y = []
            for k in epc_baseline[:, 2]:
                prob_attack = sum(
                    1 for i in spoof_eval_neg if i >= k
                ) / float(spoof_eval_neg.size)
                mix_prob_y.append(100. * prob_attack)

            mpl.gca().set_axisbelow(True)
            prob_ax = mpl.gca().twinx()
            mpl.plot(
                epc_baseline[:, 0],
                mix_prob_y,
                color='C3',
                linestyle='-',
                label=self._label(
                    'IAPMR', '%s-%s' % (dev_file[0], eval_file[0]), idx
                )
            )
            prob_ax.set_yticklabels(prob_ax.get_yticks())
            prob_ax.tick_params(axis='y', colors='red')
            prob_ax.yaxis.label.set_color('red')
            prob_ax.spines['right'].set_color('red')
            ylabels = prob_ax.get_yticks()
            prob_ax.yaxis.set_ticklabels(["%.0f" % val for val in ylabels])
            prob_ax.set_axisbelow(True)
        title = self._titles[idx] if self._titles is not None else self._title
        mpl.title(title)
        #legends for all axes
        self._plot_legends()
        mpl.gcf().set_tight_layout(True)
        mpl.xticks(rotation=self._x_rotation)
        self._pdf_page.savefig(mpl.gcf())

class Epsc(PadPlot):
    ''' Handles the plotting of EPSC '''
    def __init__(self, ctx, scores, evaluation, func_load,
                 criteria, var_param, fixed_param):
        super(Epsc, self).__init__(ctx, scores, evaluation, func_load)
        if 'eval_scores_0' not in self._ctx.meta:
            raise click.UsageError("EPC requires dev and eval score files")
        self._iapmr = False if 'iapmr' not in self._ctx.meta else \
                self._ctx.meta['iapmr']
        self._wer = True if 'wer' not in self._ctx.meta else \
                self._ctx.meta['wer']
        self._criteria = 'eer' if criteria is None else criteria
        self._var_param = "omega" if var_param is None else var_param
        self._fixed_param = 0.5 if fixed_param is None else fixed_param
        self._title = ''
        self._eval = True #always eval data with EPC
        self._split = False
        self._nb_figs = 1

    def compute(self, idx, dev_score, dev_file, eval_score, eval_file=None):
        ''' Plot EPSC for PAD'''
        licit_dev_neg = dev_score[0][0]
        licit_dev_pos = dev_score[0][1]
        licit_eval_neg = eval_score[0][0]
        licit_eval_pos = eval_score[0][1]
        spoof_dev_neg = dev_score[1][0]
        spoof_dev_pos = dev_score[1][1]
        spoof_eval_neg = eval_score[1][0]
        spoof_eval_pos = eval_score[1][1]
        title = self._titles[idx] if self._titles is not None else None

        mpl.gcf().clear()
        points = 10

        if self._var_param == 'omega':
            omega, beta, thrs = error_utils.epsc_thresholds(
                licit_dev_neg,
                licit_dev_pos,
                spoof_dev_neg,
                spoof_dev_pos,
                points=points,
                criteria=self._criteria,
                beta=self._fixed_param)
        else:
            omega, beta, thrs = error_utils.epsc_thresholds(
                licit_dev_neg,
                licit_dev_pos,
                spoof_dev_neg,
                spoof_dev_pos,
                points=points,
                criteria= self._criteria,
                omega=self._fixed_param
            )

        errors = error_utils.all_error_rates(
            licit_eval_neg, licit_eval_pos, spoof_eval_neg,
            spoof_eval_pos, thrs, omega, beta
        )  # error rates are returned in a list in the
           # following order: frr, far, IAPMR, far_w, wer_w

        ax1 = mpl.subplot(
            111
        )  # EPC like curves for FVAS fused scores for weighted error rates
           # between the negatives (impostors and Presentation attacks)
        if self._wer:
            if self._var_param == 'omega':
                mpl.plot(
                    omega,
                    100. * errors[4].flatten(),
                    color='C0',
                    linestyle='-',
                    label=r"WER$_{\omega,\beta}$")
                mpl.xlabel(r"Weight $\omega$")
            else:
                mpl.plot(
                    beta,
                    100. * errors[4].flatten(),
                    color='C0',
                    linestyle='-',
                    label=r"WER$_{\omega,\beta}$")
                mpl.xlabel(r"Weight $\beta$")
            mpl.ylabel(r"WER$_{\omega,\beta}$ (%)")

        if self._iapmr:
            axis = mpl.gca()
            if self._wer:
                axis = mpl.twinx()
                axis.grid(False)
            if self._var_param == 'omega':
                mpl.plot(
                    omega,
                    100. * errors[2].flatten(),
                    color='C3',
                    linestyle='-',
                    label='IAPMR')
                mpl.xlabel(r"Weight $\omega$")
            else:
                mpl.plot(
                    beta,
                    100. * errors[2].flatten(),
                    color='C3',
                    linestyle='-',
                    label='IAPMR')
                mpl.xlabel(r"Weight $\beta$")
            mpl.ylabel(r"IAPMR  (%)")
            if self._wer:
                axis.set_yticklabels(axis.get_yticks())
                axis.tick_params(axis='y', colors='red')
                axis.yaxis.label.set_color('red')
                axis.spines['right'].set_color('red')

        if self._var_param == 'omega':
            mpl.title(r"EPSC with $\beta$ = %.2f" % (
                self._fixed_param,) if title is None else title)
        else:
            mpl.title(r"EPSC with $\omega$ = %.2f" % (
                self._fixed_param,) if title is None else title)

        mpl.grid()
        self._plot_legends()
        ax1.set_xticklabels(ax1.get_xticks())
        ax1.set_yticklabels(ax1.get_yticks())
        mpl.xticks(rotation=self._x_rotation)
        self._pdf_page.savefig(bbox_inches='tight')

class Epsc3D(Epsc):
    ''' 3D EPSC plots for PAD'''
    def compute(self, idx, dev_score, dev_file, eval_score, eval_file=None):
        ''' Implements plots'''
        licit_dev_neg = dev_score[0][0]
        licit_dev_pos = dev_score[0][1]
        licit_eval_neg = eval_score[0][0]
        licit_eval_pos = eval_score[0][1]
        spoof_dev_neg = dev_score[1][0]
        spoof_dev_pos = dev_score[1][1]
        spoof_eval_neg = eval_score[1][0]
        spoof_eval_pos = eval_score[1][1]

        title = self._titles[idx] if self._titles is not None else None

        mpl.gcf().clear()

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        points = 10

        omega, beta, thrs = error_utils.epsc_thresholds(
            licit_dev_neg,
            licit_dev_pos,
            spoof_dev_neg,
            spoof_dev_pos,
            points=points,
            criteria=self._criteria)

        errors = error_utils.all_error_rates(
            licit_eval_neg, licit_eval_pos, spoof_eval_neg, spoof_eval_pos,
            thrs, omega, beta
        )
        # error rates are returned in a list as 2D numpy.ndarrays in
        # the following order: frr, far, IAPMR, far_w, wer_wb, hter_wb
        wer_errors = 100 * errors[2 if self._iapmr else 4]

        ax1 = mpl.gcf().add_subplot(111, projection='3d')

        W, B = np.meshgrid(omega, beta)

        ax1.plot_wireframe(
            W, B, wer_errors, cmap=cm.coolwarm, antialiased=False
        )  # surface

        if self._iapmr:
            ax1.azim = -30
            ax1.elev = 50

        ax1.set_xlabel(r"Weight $\omega$")
        ax1.set_ylabel(r"Weight $\beta$")
        ax1.set_zlabel(
            r"WER$_{\omega,\beta}$ (%)" if self._wer else "IAPMR (%)"
        )

        mpl.title("3D EPSC" if title is None else title)

        ax1.set_xticklabels(ax1.get_xticks())
        ax1.set_yticklabels(ax1.get_yticks())
        ax1.set_zticklabels(ax1.get_zticks())

        self._pdf_page.savefig(bbox_inches='tight')
