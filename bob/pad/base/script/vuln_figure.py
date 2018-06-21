'''Runs error analysis on score sets, outputs metrics and plots'''

import math
import click
import numpy as np
import matplotlib.pyplot as mpl
import bob.measure.script.figure as measure_figure
import bob.bio.base.script.figure as bio_figure
from tabulate import tabulate
from bob.measure.utils import get_fta_list
from bob.measure import (
    frr_threshold, far_threshold, eer_threshold, min_hter_threshold, farfrr, epc, ppndf
)
from bob.measure.plot import (det, det_axis, roc_for_far, log_values)
from . import error_utils


class Metrics(measure_figure.Metrics):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Metrics, self).__init__(ctx, scores, evaluation, func_load)

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
        rows.append(['FNMR (%)', '{:>5.1f}%'.format(frr * 100)])
        rows.append(['HTER (%)', '{:>5.1f}%'.format(50 * (far + frr))])
        rows.append(['IAPMR (%)', '{:>5.1f}%'.format(100 * iapmr)])
        click.echo(
            tabulate(rows, headers, self._tablefmt),
            file=self.log_file
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

    def __init__(self, ctx, scores, evaluation, func_load):
        super(HistVuln, self).__init__(
            ctx, scores, evaluation, func_load, nhist_per_system=3)

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
        self._sampling = ctx.meta.get('sampling', 5)

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

        points = self._sampling or 5

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


class BaseVulnDetRoc(PadPlot):
    '''Base for DET and ROC'''

    def __init__(self, ctx, scores, evaluation, func_load, real_data,
                 no_spoof):
        super(BaseVulnDetRoc, self).__init__(
            ctx, scores, evaluation, func_load)
        self._no_spoof = no_spoof
        self._hlines_at = ctx.meta.get('hlines_at', [])
        self._real_data = True if real_data is None else real_data
        self._legend_loc = None

    def compute(self, idx, input_scores, input_names):
        ''' Implements plots'''
        licit_neg = input_scores[0][0]
        licit_pos = input_scores[0][1]
        spoof_neg = input_scores[1][0]
        spoof_pos = input_scores[1][1]
        self._plot(
            licit_neg,
            licit_pos,
            self._points,
            color='C0',
            linestyle='-',
            label=self._label("licit", input_names[0], idx)
        )
        if not self._no_spoof and spoof_neg is not None:
            ax1 = mpl.gca()
            ax2 = ax1.twiny()
            ax2.set_xlabel('IAPMR', color='C3')
            ax2.set_xticklabels(ax2.get_xticks())
            ax2.tick_params(axis='x', colors='C3')
            ax2.xaxis.label.set_color('C3')
            ax2.spines['top'].set_color('C3')
            ax2.spines['bottom'].set_color('C0')
            ax1.xaxis.label.set_color('C0')
            ax1.tick_params(axis='x', colors='C0')
            self._plot(
                spoof_neg,
                spoof_pos,
                self._points,
                color='C3',
                linestyle=':',
                label=self._label("spoof", input_names[1], idx)
            )
            mpl.sca(ax1)

        if self._hlines_at is None:
            return

        for line in self._hlines_at:
            thres_baseline = frr_threshold(licit_neg, licit_pos, line)

            axlim = mpl.axis()

            farfrr_licit, farfrr_licit_det = self._get_farfrr(
                licit_neg, licit_pos,
                thres_baseline
            )
            if farfrr_licit is None:
                return

            farfrr_spoof, farfrr_spoof_det = self._get_farfrr(
                spoof_neg, spoof_pos,
                frr_threshold(spoof_neg, spoof_pos, farfrr_licit[1])
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
                    ('FMNR', farfrr_licit[1] * 100))

            if not self._real_data:
                label_licit = '%s @ operating point' % self._y_label
                label_spoof = 'IAPMR @ operating point'
            else:
                label_licit = 'FMR=%.2f%%' % (farfrr_licit[0] * 100)
                label_spoof = 'IAPMR=%.2f%%' % (farfrr_spoof[0] * 100)

            mpl.plot(
                farfrr_licit_det[0],
                farfrr_licit_det[1],
                'o',
                color='C0',
                label=label_licit
            )  # FAR point, licit scenario
            mpl.plot(
                farfrr_spoof_det[0],
                farfrr_spoof_det[1],
                'o',
                color='C3',
                label=label_spoof
            )  # FAR point, spoof scenario


    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures and
        close pdf is needed '''
        # only for plots

        if self._title.replace(' ', ''):
            mpl.title(self._title, y=1.15)
        mpl.xlabel(self._x_label)
        mpl.ylabel(self._y_label)
        mpl.grid(True, color=self._grid_color)
        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            li, la = ax.get_legend_handles_labels()
            lines += li
            labels += la
            mpl.sca(ax)
            self._set_axis()
            fig = mpl.gcf()
            mpl.xticks(rotation=self._x_rotation)
            mpl.tick_params(axis='both', which='major', labelsize=6)
        if self._disp_legend:
            mpl.gca().legend(
                lines, labels, loc=self._legend_loc, fancybox=True,
                framealpha=0.5
            )
        self._pdf_page.savefig(fig)

        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
                ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    def _get_farfrr(self, x, y, thres):
        return None, None

    def _plot(self, x, y, points, **kwargs):
        pass


class DetVuln(BaseVulnDetRoc):
    '''DET for vuln'''

    def __init__(self, ctx, scores, evaluation, func_load, real_data,
                 no_spoof):
        super(Det, self).__init__(ctx, scores, evaluation, func_load,
                                  real_data, no_spoof)
        self._x_label = self._x_label or "FMR"
        self._y_label = self._y_label or "FNMR"
        add = ''
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        self._title = self._title or ('DET: LICIT' + add)
        self._legend_loc = self._legend_loc or 'upper right'

    def _set_axis(self):
        if self._axlim is not None and None not in self._axlim:
            det_axis(self._axlim)
        else:
            det_axis([0.01, 99, 0.01, 99])

    def _get_farfrr(self, x, y, thres):
        points = farfrr(x, y, thres)
        return points, [ppndf(i) for i in points]

    def _plot(self, x, y, points, **kwargs):
        det(
            x, y, points,
            color=kwargs.get('color'),
            linestyle=kwargs.get('linestyle'),
            label=kwargs.get('label')
        )


class RocVuln(BaseVulnDetRoc):
    '''ROC for vuln'''

    def __init__(self, ctx, scores, evaluation, func_load, real_data, no_spoof):
        super(RocVuln, self).__init__(ctx, scores, evaluation, func_load,
                                      real_data, no_spoof)
        self._x_label = self._x_label or "FMR"
        self._y_label = self._y_label or "1 - FNMR"
        self._semilogx = ctx.meta.get('semilogx', True)
        add = ''
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        self._title = self._title or ('ROC: LICIT' + add)
        best_legend = 'lower right' if self._semilogx else 'upper right'
        self._legend_loc = self._legend_loc or best_legend

    def _plot(self, x, y, points, **kwargs):
        roc_for_far(
            x, y,
            far_values=log_values(self._min_dig or -4),
            CAR=self._semilogx,
            color=kwargs.get('color'), linestyle=kwargs.get('linestyle'),
            label=kwargs.get('label')
        )

    def _get_farfrr(self, x, y, thres):
        points = farfrr(x, y, thres)
        points2 = (points[0], 1 - points[1])
        return points, points2


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
        mpl.xlabel(self._x_label or "FMR (%)")
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
