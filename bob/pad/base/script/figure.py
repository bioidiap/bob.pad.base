'''Runs error analysis on score sets, outputs metrics and plots'''

import click
import numpy as np
import matplotlib.pyplot as mpl
import bob.measure.script.figure as measure_figure
import bob.bio.base.script.figure as bio_figure
from tabulate import tabulate
from bob.measure.utils import get_fta_list
from bob.measure import (
    far_threshold, eer_threshold, min_hter_threshold, farfrr, epc, ppndf
)
from bob.measure.plot import (det, det_axis, roc_for_far, log_values)
from . import error_utils

ALL_CRITERIA = ('bpcer20', 'eer', 'min-hter')


def calc_threshold(method, neg, pos):
    """Calculates the threshold based on the given method.
    The scores should be sorted!

    Parameters
    ----------
    method : str
        One of ``bpcer20``, ``eer``, ``min-hter``.
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

    def compute(self, idx, input_scores, input_names):
        ''' Compute metrics for the given criteria'''
        neg_list, pos_list, _ = get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        dev_file = input_names[0]
        if self._eval:
            eval_neg, eval_pos = neg_list[1], pos_list[1]
            eval_file = input_names[1]

        title = self._legends[idx] if self._legends is not None else None
        headers = ['' or title, 'Development %s' % dev_file]
        if self._eval:
            headers.append('Eval. % s' % eval_file)
        for m in ALL_CRITERIA:
            raws = []
            threshold = calc_threshold(m, dev_neg, dev_pos)
            click.echo("\nThreshold of %f selected with the %s criteria" % (
                threshold, m))
            apcer, bpcer = farfrr(dev_neg, dev_pos, threshold)
            raws.append(['APCER', '{:>5.1f}%'.format(apcer * 100)])
            raws.append(['BP', '{:>5.1f}%'.format(bpcer * 100)])
            raws.append(['ACER', '{:>5.1f}%'.format((apcer + bpcer) * 50)])
            if self._eval and eval_neg is not None:
                apcer, bpcer = farfrr(eval_neg, eval_pos, threshold)
                raws[0].append('{:>5.1f}%'.format(apcer * 100))
                raws[1].append('{:>5.1f}%'.format(bpcer * 100))
                raws[2].append('{:>5.1f}%'.format((apcer + bpcer) * 50))

            click.echo(
                tabulate(raws, headers, self._tablefmt),
                file=self.log_file
            )


class MetricsVuln(measure_figure.Metrics):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(MetricsVuln, self).__init__(ctx, scores, evaluation, func_load)

    ''' Compute metrics from score files'''

    def compute(self, idx, input_scores, input_names):
        ''' Compute metrics for the given criteria'''
        neg_list, pos_list, _ = get_fta_list(input_scores)
        dev_neg, dev_pos = neg_list[0], pos_list[0]
        criter = self._criterion or 'eer'
        threshold = calc_threshold(criter, dev_neg, dev_pos) \
            if self._thres is None else self._thres[idx]
        far, frr = farfrr(neg_list[1], pos_list[1], threshold)
        iapmr, _ = farfrr(neg_list[3], pos_list[1], threshold)
        title = self._legends[idx] if self._legends is not None else None
        headers = ['' or title, '%s (threshold=%.2g)' %
                   (criter.upper(), threshold)]
        rows = []
        rows.append(['FMR (%)', '{:>5.1f}%'.format(100 * far)])
        rows.append(['BPCER (%)', '{:>5.1f}%'.format(frr * 100)])
        rows.append(['HTER (%)', '{:>5.1f}%'.format(50 * (far + frr))])
        rows.append(['IAPMR (%)', '{:>5.1f}%'.format(100 * iapmr)])
        click.echo(
            tabulate(rows, headers, self._tablefmt),
            file=self.log_file
        )


class HistPad(measure_figure.Hist):
    ''' Histograms for PAD '''

    def _setup_hist(self, neg, pos):
        self._title_base = 'PAD'
        self._density_hist(
            pos[0], n=0, label='Bona Fide', color='C1'
        )
        self._density_hist(
            neg[0], n=1, label='Presentation attack', alpha=0.4, color='C7',
            hatch='\\\\'
        )


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
        mix_prob_y.append(100. * error_utils.calc_pass_rate(k, scores))

    mpl.plot(thres, mix_prob_y, label='IAPMR', color='C3', **kwargs)


def _iapmr_plot(scores, threshold, iapmr, real_data, **kwargs):
    _iapmr_dot(threshold, iapmr, real_data, **kwargs)
    _iapmr_line_plot(scores, n_points=100, **kwargs)


class HistVuln(measure_figure.Hist):
    ''' Histograms for vulnerability '''

    def _setup_hist(self, neg, pos):
        self._title_base = 'Vulnerability'
        self._density_hist(
            pos[0], n=0, label='Genuine', color='C2'
        )
        self._density_hist(
            neg[0], n=1, label='Zero-effort impostors', alpha=0.8, color='C0'
        )
        self._density_hist(
            neg[1], n=2, label='Presentation attack', alpha=0.4, color='C7',
            hatch='\\\\'
        )

    def _lines(self, threshold, label, neg, pos, idx, **kwargs):
        if 'iapmr_line' not in self._ctx.meta or self._ctx.meta['iapmr_line']:
            # plot vertical line
            super(HistVuln, self)._lines(threshold, label, neg, pos, idx)

            # plot iapmr_line
            iapmr, _ = farfrr(neg[1], pos[0], threshold)
            ax2 = mpl.twinx()
            # we never want grid lines on axis 2
            ax2.grid(False)
            real_data = True if 'real_data' not in self._ctx.meta else \
                self._ctx.meta['real_data']
            _iapmr_plot(neg[1], threshold, iapmr, real_data=real_data)
            n = idx % self._step_print
            col = n % self._ncols
            rest_print = self.n_systems - \
                int(idx / self._step_print) * self._step_print
            if col == self._ncols - 1 or n == rest_print - 1:
                ax2.set_ylabel("IAPMR (%)", color='C3')
            ax2.tick_params(axis='y', colors='red')
            ax2.yaxis.label.set_color('red')
            ax2.spines['right'].set_color('red')


class PadPlot(measure_figure.PlotBase):
    '''Base class for PAD plots'''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(PadPlot, self).__init__(ctx, scores, evaluation, func_load)
        mpl.rcParams['figure.constrained_layout.use'] = self._clayout

    def end_process(self):
        '''Close pdf '''
        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
           ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    def _plot_legends(self):
        # legends for all axes
        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            li, la = ax.get_legend_handles_labels()
            lines += li
            labels += la
        if self._disp_legend:
            mpl.gca().legend(lines, labels, loc=self._legend_loc,
                             fancybox=True, framealpha=0.5)


class Epc(PadPlot):
    ''' Handles the plotting of EPC '''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Epc, self).__init__(ctx, scores, evaluation, func_load)
        self._iapmr = True if 'iapmr' not in self._ctx.meta else \
            self._ctx.meta['iapmr']
        self._title = self._title or ('EPC and IAPMR' if self._iapmr else
                                      'EPC')
        self._x_label = self._x_label or r"Weight $\beta$"
        self._y_label = self._y_label or "WER (%)"
        self._eval = True  # always eval data with EPC
        self._split = False
        self._nb_figs = 1

        if self._min_arg != 4:
            raise click.BadParameter("You must provide 4 scores files:{licit,"
                                     "spoof}/{dev,eval}")

    def compute(self, idx, input_scores, input_names):
        ''' Plot EPC for PAD'''
        licit_dev_neg = input_scores[0][0]
        licit_dev_pos = input_scores[0][1]
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_eval_neg = input_scores[3][0]
        mpl.gcf().clear()
        epc_baseline = epc(
            licit_dev_neg, licit_dev_pos, licit_eval_neg,
            licit_eval_pos, 100
        )
        mpl.plot(
            epc_baseline[:, 0], [100. * k for k in epc_baseline[:, 1]],
            color='C0',
            label=self._label(
                'WER', '%s-%s' % (input_names[0], input_names[1]), idx
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
                    'IAPMR', '%s-%s' % (input_names[0], input_names[1]), idx
                )
            )
            prob_ax.set_yticklabels(prob_ax.get_yticks())
            prob_ax.tick_params(axis='y', colors='C3')
            prob_ax.yaxis.label.set_color('C3')
            prob_ax.spines['right'].set_color('C3')
            ylabels = prob_ax.get_yticks()
            prob_ax.yaxis.set_ticklabels(["%.0f" % val for val in ylabels])
            prob_ax.set_ylabel('IAPMR', color='C3')
            prob_ax.set_axisbelow(True)
        title = self._legends[idx] if self._legends is not None else self._title
        if title.replace(' ', ''):
            mpl.title(title)
        # legends for all axes
        self._plot_legends()
        mpl.xticks(rotation=self._x_rotation)
        self._pdf_page.savefig(mpl.gcf())


class Epsc(PadPlot):
    ''' Handles the plotting of EPSC '''

    def __init__(self, ctx, scores, evaluation, func_load,
                 criteria, var_param, fixed_param):
        super(Epsc, self).__init__(ctx, scores, evaluation, func_load)
        self._iapmr = False if 'iapmr' not in self._ctx.meta else \
            self._ctx.meta['iapmr']
        self._wer = True if 'wer' not in self._ctx.meta else \
            self._ctx.meta['wer']
        self._criteria = 'eer' if criteria is None else criteria
        self._var_param = "omega" if var_param is None else var_param
        self._fixed_param = 0.5 if fixed_param is None else fixed_param
        self._eval = True  # always eval data with EPC
        self._split = False
        self._nb_figs = 1
        self._title = ''

        if self._min_arg != 4:
            raise click.BadParameter("You must provide 4 scores files:{licit,"
                                     "spoof}/{dev,eval}")

    def compute(self, idx, input_scores, input_names):
        ''' Plot EPSC for PAD'''
        licit_dev_neg = input_scores[0][0]
        licit_dev_pos = input_scores[0][1]
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_dev_neg = input_scores[2][0]
        spoof_dev_pos = input_scores[2][1]
        spoof_eval_neg = input_scores[3][0]
        spoof_eval_pos = input_scores[3][1]
        title = self._legends[idx] if self._legends is not None else None

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
                criteria=self._criteria,
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
                mpl.xlabel(self._x_label or r"Weight $\omega$")
            else:
                mpl.plot(
                    beta,
                    100. * errors[4].flatten(),
                    color='C0',
                    linestyle='-',
                    label=r"WER$_{\omega,\beta}$")
                mpl.xlabel(self._x_label or r"Weight $\beta$")
            mpl.ylabel(self._y_label or r"WER$_{\omega,\beta}$ (%)")

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
                mpl.xlabel(self._x_label or r"Weight $\omega$")
            else:
                mpl.plot(
                    beta,
                    100. * errors[2].flatten(),
                    color='C3',
                    linestyle='-',
                    label='IAPMR')
                mpl.xlabel(self._x_label or r"Weight $\beta$")
            mpl.ylabel(self._y_label or r"IAPMR  (%)")
            if self._wer:
                axis.set_yticklabels(axis.get_yticks())
                axis.tick_params(axis='y', colors='red')
                axis.yaxis.label.set_color('red')
                axis.spines['right'].set_color('red')

        if self._var_param == 'omega':
            if title is not None and title.replace(' ', ''):
                mpl.title(title or (r"EPSC with $\beta$ = %.2f" %
                                    self._fixed_param))
        else:
            if title is not None and title.replace(' ', ''):
                mpl.title(title or (r"EPSC with $\omega$ = %.2f" %
                                    self._fixed_param))

        mpl.grid()
        self._plot_legends()
        ax1.set_xticklabels(ax1.get_xticks())
        ax1.set_yticklabels(ax1.get_yticks())
        mpl.xticks(rotation=self._x_rotation)
        self._pdf_page.savefig()


class Epsc3D(Epsc):
    ''' 3D EPSC plots for PAD'''

    def compute(self, idx, input_scores, input_names):
        ''' Implements plots'''
        licit_dev_neg = input_scores[0][0]
        licit_dev_pos = input_scores[0][1]
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_dev_neg = input_scores[2][0]
        spoof_dev_pos = input_scores[2][1]
        spoof_eval_neg = input_scores[3][0]
        spoof_eval_pos = input_scores[3][1]

        title = self._legends[idx] if self._legends is not None else "3D EPSC"

        mpl.rcParams.pop('key', None)

        mpl.gcf().clear()
        mpl.gcf().set_constrained_layout(self._clayout)

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

        ax1.set_xlabel(self._x_label or r"Weight $\omega$")
        ax1.set_ylabel(self._y_label or r"Weight $\beta$")
        ax1.set_zlabel(
            r"WER$_{\omega,\beta}$ (%)" if self._wer else "IAPMR (%)"
        )

        if title.replace(' ', ''):
            mpl.title(title)

        ax1.set_xticklabels(ax1.get_xticks())
        ax1.set_yticklabels(ax1.get_yticks())
        ax1.set_zticklabels(ax1.get_zticks())

        self._pdf_page.savefig()


class Roc(bio_figure.Roc):
    '''ROC for PARD'''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Roc, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = ctx.meta.get('x_label') or 'APCER'
        self._y_label = ctx.meta.get('y_label') or '1 - BPCER'

class DetPad(bio_figure.Det):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(DetPad, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = ctx.meta.get('x_label') or 'APCER'
        self._y_label = ctx.meta.get('y_label') or 'BPCER'



class BaseDetRoc(PadPlot):
    '''Base for DET and ROC'''

    def __init__(self, ctx, scores, evaluation, func_load, criteria, real_data,
                no_spoof):
        super(BaseDetRoc, self).__init__(ctx, scores, evaluation, func_load)
        self._no_spoof = no_spoof
        self._criteria = criteria or 'eer'
        self._real_data = True if real_data is None else real_data


    def compute(self, idx, input_scores, input_names):
        ''' Implements plots'''
        licit_dev_neg = input_scores[0][0]
        licit_dev_pos = input_scores[0][1]
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_eval_neg = input_scores[3][0] if len(input_scores) > 2 else None
        spoof_eval_pos = input_scores[3][1] if len(input_scores) > 2 else None
        self._plot(
            licit_eval_neg,
            licit_eval_pos,
            self._points,
            color='C0',
            linestyle='-',
            label=self._label("licit", input_names[0], idx)
        )
        if not self._no_spoof and spoof_eval_neg is not None:
            self._plot(
                spoof_eval_neg,
                spoof_eval_pos,
                self._points,
                color='C3',
                linestyle=':',
                label=self._label("spoof", input_names[3], idx)
            )

        if self._criteria is None or self._no_spoof:
            return

        thres_baseline = calc_threshold(
            self._criteria, licit_dev_neg, licit_dev_pos
        )

        axlim = mpl.axis()

        farfrr_licit, farfrr_licit_det = self._get_farfrr(
            licit_eval_neg, licit_eval_pos,
            thres_baseline
        )
        if farfrr_licit is None:
            return
        farfrr_spoof, farfrr_spoof_det = self._get_farfrr(
            spoof_eval_neg, spoof_eval_pos, thres_baseline
        )
        if not self._real_data:
            mpl.axhline(
                y=farfrr_licit_det[1],
                xmin=axlim[2],
                xmax=axlim[3],
                color='k',
                linestyle='--',
                label="%s @ EER" % self._y_label)
        else:
            mpl.axhline(
                y=farfrr_licit_det[1],
                xmin=axlim[0],
                xmax=axlim[1],
                color='k',
                linestyle='--',
                label="%s = %.2f%%" %
                (self._y_label, farfrr_licit[1] * 100))

        mpl.plot(
            farfrr_licit_det[0],
            farfrr_licit_det[1],
            'o',
            color='C0',
        )  # FAR point, licit scenario
        mpl.plot(
            farfrr_spoof_det[0],
            farfrr_spoof_det[1],
            'o',
            color='C3',
        )  # FAR point, spoof scenario

        # annotate the FAR points
        xyannotate_licit = [
            0.15 + farfrr_licit_det[0],
            farfrr_licit_det[1] - 0.15,
        ]
        xyannotate_spoof = [
            0.15 + farfrr_spoof_det[0],
            farfrr_spoof_det[1] - 0.15,
        ]

        if not self._real_data:
            mpl.annotate(
                '%s @ operating point' % self._y_label,
                xy=(farfrr_licit_det[0], farfrr_licit_det[1]),
                xycoords='data',
                xytext=(xyannotate_licit[0], xyannotate_licit[1]))
            mpl.annotate(
                'IAPMR @ operating point',
                xy=(farfrr_spoof_det[0], farfrr_spoof_det[1]),
                xycoords='data',
                xytext=(xyannotate_spoof[0], xyannotate_spoof[1]))
        else:
            mpl.annotate(
                'APCER=%.2f%%' % (farfrr_licit[0] * 100),
                xy=(farfrr_licit_det[0], farfrr_licit_det[1]),
                xycoords='data',
                xytext=(xyannotate_licit[0], xyannotate_licit[1]),
                color='C0',
                size='small')
            mpl.annotate(
                'IAPMR=%.2f%%' % (farfrr_spoof[0] * 100),
                xy=(farfrr_spoof_det[0], farfrr_spoof_det[1]),
                xycoords='data',
                xytext=(xyannotate_spoof[0], xyannotate_spoof[1]),
                color='C3',
                size='small')

    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures and
        close pdf is needed '''
        # only for plots

        if self._title.replace(' ', ''):
            mpl.title(self._title)
        mpl.xlabel(self._x_label)
        mpl.ylabel(self._y_label)
        mpl.grid(True, color=self._grid_color)
        if self._disp_legend:
            mpl.legend(loc=self._legend_loc)
        self._set_axis()
        fig = mpl.gcf()
        mpl.xticks(rotation=self._x_rotation)
        mpl.tick_params(axis='both', which='major', labelsize=4)
        for tick in mpl.gca().xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in mpl.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(6)

        self._pdf_page.savefig(fig)

        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
                ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    def _get_farfrr(self, x, y, thres):
        return None, None

    def _plot(self, x, y, points, **kwargs):
        pass

class Det(BaseDetRoc):
    '''Base for DET and ROC'''

    def __init__(self, ctx, scores, evaluation, func_load, criteria, real_data,
                no_spoof):
        super(Det, self).__init__(ctx, scores, evaluation, func_load,
                                  criteria, real_data, no_spoof)
        self._x_label = self._x_label or "APCER"
        self._y_label = self._y_label or "BPCER"
        add = ''
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        self._title = self._title or ('DET: LICIT' + add)


    def _set_axis(self):
        if self._axlim is not None and None not in self._axlim:
            det_axis(self._axlim)
        else:
            det_axis([0.01, 99, 0.01, 99])

    def _get_farfrr(self, x, y, thres):
        # calculate test frr @ EER (licit scenario)
        points = farfrr(x, y, thres)
        return points, [ppndf(i) for i in points]


    def _plot(self, x, y, points, **kwargs):
        det(
            x, y, points,
            color=kwargs.get('color'),
            linestyle=kwargs.get('linestyle'),
            label=kwargs.get('label')
        )

class RocVuln(BaseDetRoc):
    '''ROC for vuln'''

    def __init__(self, ctx, scores, evaluation, func_load, criteria, real_data,
                no_spoof):
        super(RocVuln, self).__init__(ctx, scores, evaluation, func_load,
                                      criteria, real_data, no_spoof)
        self._x_label = self._x_label or "APCER"
        self._y_label = self._y_label or "1 - BPCER"
        self._semilogx = ctx.meta.get('semilogx', True)
        add = ''
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        self._title = self._title or ('ROC: LICIT' + add)


    def _plot(self, x, y, points, **kwargs):
        roc_for_far(
            x, y,
            far_values=log_values(self._min_dig or -4),
            CAR=self._semilogx,
            color=kwargs.get('color'), linestyle=kwargs.get('linestyle'),
            label=kwargs.get('label')
        )


class FmrIapmr(PadPlot):
    '''FMR vs IAPMR'''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(FmrIapmr, self).__init__(ctx, scores, evaluation, func_load)
        self._eval = True  # always eval data with EPC
        self._split = False
        self._nb_figs = 1
        self._semilogx = False if 'semilogx' not in ctx.meta else\
            ctx.meta['semilogx']
        if self._min_arg != 4:
            raise click.BadParameter("You must provide 4 scores files:{licit,"
                                     "spoof}/{dev,eval}")

    def compute(self, idx, input_scores, input_names):
        ''' Implements plots'''
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_eval_neg = input_scores[3][0]
        fmr_list = np.linspace(0, 1, 100)
        iapmr_list = []
        for i, fmr in enumerate(fmr_list):
            thr = far_threshold(licit_eval_neg, licit_eval_pos, fmr, True)
            iapmr_list.append(farfrr(spoof_eval_neg, licit_eval_pos, thr)[0])
            # re-calculate fmr since threshold might give a different result
            # for fmr.
            fmr_list[i] = farfrr(licit_eval_neg, licit_eval_pos, thr)[0]
        label = self._legends[idx] if self._legends is not None else \
            '(%s/%s)' % (input_names[1], input_names[3])
        if self._semilogx:
            mpl.semilogx(fmr_list, iapmr_list, label=label)
        else:
            mpl.plot(fmr_list, iapmr_list, label=label)

    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures and
        close pdf is needed '''
        # only for plots
        title = self._title if self._title is not None else "FMR vs IAPMR"
        if title.replace(' ', ''):
            mpl.title(title)
        mpl.xlabel(self._x_label or "False Match Rate (%)")
        mpl.ylabel(self._y_label or "IAPMR (%)")
        mpl.grid(True, color=self._grid_color)
        if self._disp_legend:
            mpl.legend(loc=self._legend_loc)
        self._set_axis()
        fig = mpl.gcf()
        mpl.xticks(rotation=self._x_rotation)
        mpl.tick_params(axis='both', which='major', labelsize=4)

        self._pdf_page.savefig(fig)

        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
                ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()
