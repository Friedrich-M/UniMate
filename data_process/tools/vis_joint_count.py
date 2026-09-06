"""Plot the joints-per-skeleton distribution of ``joint_count.json`` files.

Stage 1 writes ``<export_dir>/joint_count.json`` — a flat ``{object_type:
n_joints}`` map (see :func:`data_process.utils.blender_export.write_export_summary`),
one entry per exported skeleton after pruning. This tool turns one or more
of those files into paper-style distribution figures plus a summary table,
so the joint-count filters — stage 4's ``--min_joints`` / ``--max_joints``
and the training ``dataset.max_joints`` cap — can be checked against the
data instead of guessed.

Read-only: the JSON inputs are never modified. Styling follows the other
dataset-statistics figures (see :mod:`data_process.utils.dist_plot`).

Outputs:

  * ``<export_dir>/joint_count_distribution.png`` — per file, written next
    to its JSON (PNG only) so every export directory carries its own
    joint-count figure. Two panels:
      - histogram of joints per skeleton as a proportion of the file's
        skeletons, integer-aligned bins (width 1 / 2 / 3 by span), with a
        kernel density estimate and mean / median guides;
      - the cumulative curve "share of skeletons with at most N joints" —
        the fraction a ``max_joints`` cap keeps — with the threshold joint
        counts marked and the share labeled at each.
    ``--per_dataset_dir DIR`` collects them in one directory instead, as
    ``DIR/<label>_joint_count_distribution.{png,pdf}``.
  * ``<output_dir>/joint_count_distribution_comparison.{png,pdf}`` — when
    several files are given: the kernel densities and cumulative curves of
    every dataset overlaid, each normalized to its own skeleton count.
  * ``<output_dir>/joint_count_distribution_pooled.{png,pdf}`` — with
    ``--pooled``: every input's skeletons in one figure (the layout of the
    paper's joint-count figure).
  * ``<output_dir>/joint_count_summary.json`` — the numbers behind the plots
    (percentiles, per-threshold counts and shares), also printed as a table.

A single-skeleton file (mixamo) still gets its figure — one bar, no
density curve — so every export directory is covered.

Usage:
    python -m data_process.tools.vis_joint_count                        # every dataset/export/*/joint_count.json
    python -m data_process.tools.vis_joint_count dataset/export/objaverse
    python -m data_process.tools.vis_joint_count dataset/export/{truebones,objaverse} \\
        --thresholds 65 --clip_max 90 --pooled
"""

import argparse
import json
import os

import numpy as np

from data_process.utils import dist_plot as dp


DEFAULT_GLOB = 'dataset/export/*/joint_count.json'
DEFAULT_OUTPUT_DIR = 'outputs/joint_count_vis'
FILENAME = 'joint_count.json'
SUFFIX = 'distribution'
# Stage-4 default --min_joints, the training dataset.max_joints cap, and the
# plotting cap the paper's joint-count figure used.
DEFAULT_THRESHOLDS = '8,65,90'


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summarize(joints, thresholds):
    stats = {'n_skeletons': int(joints.size)}
    stats.update(dp.percentile_stats(joints))
    # Share of skeletons with at most N joints: what a max_joints cap of N
    # keeps (1 - share is what it drops).
    stats['at_most'] = {
        str(t): {'count': int((joints <= t).sum()),
                 'share': float((joints <= t).mean())}
        for t in thresholds
    }
    return stats


def print_table(stats_by_label, thresholds):
    cols = ['skeletons', 'min', 'p10', 'median', 'mean', 'p90', 'max']
    cols += [f'<={t}j' for t in thresholds]
    width = max(len(lab) for lab in stats_by_label)
    print(f'{"dataset":<{width}}  ' + '  '.join(f'{c:>9}' for c in cols))
    for lab, s in stats_by_label.items():
        row = [f'{s["n_skeletons"]:,}', f'{s["min"]}', f'{s["p10"]:.0f}',
               f'{s["median"]:g}', f'{s["mean"]:.1f}', f'{s["p90"]:.0f}', f'{s["max"]}']
        row += [f'{s["at_most"][str(t)]["share"]:.1%}' for t in thresholds]
        print(f'{lab:<{width}}  ' + '  '.join(f'{c:>9}' for c in row))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _apply_clip(joints, clip_max):
    """Values shown in a figure and a subtitle note for the ones dropped."""
    if clip_max is None or not (joints > clip_max).any():
        return joints, ''
    kept = joints[joints <= clip_max]
    if kept.size == 0:
        raise ValueError(f'--clip_max {clip_max} drops every skeleton')
    return kept, f'  ·  {joints.size - kept.size} above {clip_max} joints not shown'


def _joint_bins(joints, bin_width, log_x):
    """Integer-aligned bins on a linear axis; even log-spaced bins otherwise.
    Returns ``(edges, bin_width_in_axis_units, lo, hi)``."""
    if log_x:
        lo, hi = dp.value_range(joints, log_x=True)
        edges = dp.even_bins(lo, hi, bin_width or 40, log_x=True)
        return edges, float(np.diff(dp.to_axis(edges[:2], True))[0]), lo, hi
    edges, width = dp.integer_bins(joints, bin_width)
    return edges, float(width), edges[0] - 0.5, edges[-1] + 0.5


def plot_dataset(label, joints, stats, thresholds, bin_width, log_x, clip_max,
                 out_base, pdf, title=None):
    """Histogram + KDE + mean/median guides, and the cumulative curve with
    threshold guides, for one joint_count.json (or a pooled set)."""
    shown_joints, clip_note = _apply_clip(joints, clip_max)
    fig, (ax_h, ax_c) = dp.two_panel_figure()
    edges, bin_width_axis, lo, hi = _joint_bins(shown_joints, bin_width, log_x)
    shown = [t for t in thresholds if lo <= t <= hi]

    # ---- Histogram (proportion of skeletons) + kernel density ----
    counts, _ = np.histogram(shown_joints, bins=edges)
    props = counts / shown_joints.size
    widths = np.diff(edges)
    ax_h.bar(edges[:-1] + 0.04 * widths, props, width=0.92 * widths, align='edge',
             color=dp.ACCENT, edgecolor='white', linewidth=0.6, alpha=0.92,
             label='Histogram', zorder=2)
    grid = dp.kde_grid(lo, hi, log_x)
    kde = dp.kde_proportion(shown_joints, grid, bin_width_axis, log_x)
    y_top = props.max()
    if kde is not None:
        ax_h.fill_between(grid, kde, color=dp.ACCENT_FL, alpha=0.30, linewidth=0, zorder=1)
        ax_h.plot(grid, kde, color=dp.ACCENT_DK, linewidth=1.8, solid_capstyle='round',
                  label='Kernel density', zorder=3)
        y_top = max(y_top, float(kde.max()))
    ax_h.set_ylim(0, y_top * dp.HEADROOM)
    dp.style_axes(ax_h)
    dp.value_axis(ax_h, lo, hi, log_x,
                  'Number of joints' + (' (log scale)' if log_x else ''))
    dp.pct_axis(ax_h)
    ax_h.set_ylabel('Proportion of skeletons')
    mean_v, median_v = float(shown_joints.mean()), float(np.median(shown_joints))
    dp.mean_median_guides(ax_h, mean_v, median_v)
    ax_h.legend(loc=dp.legend_loc_clear_of(ax_h, (mean_v, median_v)),
                borderaxespad=0.6, handlelength=1.8)

    # ---- Cumulative: share of skeletons with at most N joints ----
    xs, ys, where = dp.cdf_curve(shown_joints, span=(lo, hi))
    ax_c.fill_between(xs, ys, step=where, color=dp.ACCENT_FL, alpha=0.30,
                      linewidth=0, zorder=1)
    ax_c.step(xs, ys, where=where, color=dp.ACCENT_DK, linewidth=1.8,
              solid_capstyle='round', solid_joinstyle='round', zorder=3)
    ax_c.set_ylim(0, 1.06)
    dp.style_axes(ax_c)
    dp.value_axis(ax_c, lo, hi, log_x,
                  'Number of joints' + (' (log scale)' if log_x else ''))
    dp.pct_axis(ax_c)
    ax_c.set_ylabel('Share of skeletons with ≤ N joints')
    shares = {t: float((shown_joints <= t).mean()) for t in shown}
    dp.threshold_guides(ax_c, shown, lambda t: f'{t} joints',
                        at_bottom=dp.crowded_top(shares.get, shown))
    dp.mark_points(ax_c, [(t, shares[t]) for t in shown], prefer='below')

    subtitle = (f'{stats["n_skeletons"]:,} skeletons  ·  median {stats["median"]:g} joints'
                f'  ·  range {stats["min"]}–{stats["max"]}' + clip_note)
    return dp.finish_figure(
        fig, title or f'Distribution of Joint Counts — {dp.display_name(label)}',
        subtitle, out_base, pdf)


def plot_comparison(datasets, thresholds, bin_width, log_x, clip_max, out_base, pdf):
    """Overlay several datasets: kernel densities (proportion per bin) and
    cumulative curves, each normalized to its own skeleton count.

    ``datasets`` is ``[(label, joints), ...]``; one categorical hue per
    dataset, in input order. Single-skeleton datasets contribute a step to
    the cumulative panel but no density curve.
    """
    shown_sets = [(lab, _apply_clip(j, clip_max)[0]) for lab, j in datasets]
    all_joints = np.concatenate([j for _, j in shown_sets])
    fig, (ax_k, ax_c) = dp.two_panel_figure()
    edges, bin_width_axis, lo, hi = _joint_bins(all_joints, bin_width, log_x)
    shown = [t for t in thresholds if lo <= t <= hi]
    grid = dp.kde_grid(lo, hi, log_x)

    y_top = 0.0
    for i, (label, joints) in enumerate(shown_sets):
        color = dp.SERIES[i]
        name = dp.display_name(label)
        kde = dp.kde_proportion(joints, grid, bin_width_axis, log_x)
        if kde is not None:
            ax_k.fill_between(grid, kde, color=color, alpha=0.12, linewidth=0, zorder=1)
            ax_k.plot(grid, kde, color=color, linewidth=1.8, solid_capstyle='round',
                      label=name, zorder=3)
            y_top = max(y_top, float(kde.max()))
        xs, ys, where = dp.cdf_curve(joints, span=(lo, hi))
        ax_c.step(xs, ys, where=where, color=color, linewidth=1.8, label=name,
                  solid_capstyle='round', solid_joinstyle='round', zorder=3)

    ax_k.set_ylim(0, (y_top or 1.0) * dp.HEADROOM)
    dp.style_axes(ax_k)
    dp.value_axis(ax_k, lo, hi, log_x, 'Number of joints' + (' (log scale)' if log_x else ''))
    dp.pct_axis(ax_k)
    ax_k.set_ylabel(f'Proportion of skeletons per bin (width {bin_width_axis:g})')

    ax_c.set_ylim(0, 1.06)
    dp.style_axes(ax_c)
    dp.value_axis(ax_c, lo, hi, log_x, 'Number of joints' + (' (log scale)' if log_x else ''))
    dp.pct_axis(ax_c)
    ax_c.set_ylabel('Share of skeletons with ≤ N joints')
    crowded = dp.crowded_top(
        lambda t: max(float((j <= t).mean()) for _, j in shown_sets), shown)
    dp.threshold_guides(ax_c, shown, lambda t: f'{t} joints', at_bottom=crowded)
    # Every dataset has a cumulative step (a density needs >= 2 distinct
    # values), so the legend lives on this panel.
    ax_c.legend(loc='lower right', borderaxespad=0.6, handlelength=1.8)

    n_total = sum(j.size for _, j in datasets)
    subtitle = (f'{len(datasets)} datasets  ·  {n_total:,} skeletons  ·  '
                + '  ·  '.join(f'{dp.display_name(lab)} {j.size:,}' for lab, j in datasets))
    return dp.finish_figure(fig, 'Distribution of Joint Counts across Datasets',
                            subtitle, out_base, pdf)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot the joints-per-skeleton distribution of export-stage '
                    'joint_count.json files.')
    parser.add_argument('inputs', nargs='*',
                        help='Export directories (containing joint_count.json) or '
                             f'joint_count*.json paths. Default: {DEFAULT_GLOB}')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help='Where the comparison / pooled figures and the summary '
                             f'JSON go (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--per_dataset_dir', default=None,
                        help='Collect the per-dataset figures in this directory as '
                             '<label>_joint_count_distribution.{png,pdf}. Default: '
                             'write each one as joint_count_distribution.png next to '
                             'its own joint_count.json.')
    parser.add_argument('--thresholds', default=DEFAULT_THRESHOLDS,
                        help='Comma-separated joint counts to mark on the cumulative '
                             f'panel (default: {DEFAULT_THRESHOLDS} — stage-4 default '
                             'min_joints, the training max_joints cap, the paper '
                             'figure\'s plotting cap). Empty string disables.')
    parser.add_argument('--clip_max', type=int, default=None,
                        help='Hide skeletons with more joints than this from the '
                             'figures (the count dropped is noted in the subtitle; '
                             'the summary always covers every skeleton).')
    parser.add_argument('--bin_width', type=int, default=None,
                        help='Histogram bin width in joints (default: 1 / 2 / 3 by '
                             'value span); bin COUNT on a log axis (default 40).')
    parser.add_argument('--log', action='store_true',
                        help='Log-scaled joint axis instead of linear')
    parser.add_argument('--pooled', action='store_true',
                        help='Also plot every input\'s skeletons pooled into one figure')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Skip the overlaid comparison figure')
    parser.add_argument('--pdf', action=argparse.BooleanOptionalAction, default=True,
                        help='Also write a PDF next to the comparison / pooled figures, '
                             'and next to the per-dataset figures when --per_dataset_dir '
                             'is set (default: on; figures beside the JSONs are PNG only)')
    return parser.parse_args()


def main():
    args = parse_args()
    dp.configure_style()
    thresholds = dp.parse_int_list(args.thresholds, '--thresholds')
    paths = dp.resolve_inputs(args.inputs, FILENAME, DEFAULT_GLOB)
    labels = dp.unique_labels(paths, FILENAME)
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = []
    summary = {}
    for path, label in zip(paths, labels):
        joints = dp.load_int_map(path)
        stats = summarize(joints, thresholds)
        out_base, pdf_ok = dp.per_dataset_out_base(path, label, args.per_dataset_dir, SUFFIX)
        written = plot_dataset(label, joints, stats, thresholds, args.bin_width,
                               args.log, args.clip_max, out_base, pdf_ok and args.pdf)
        print(f'[{label}] {joints.size:,} skeletons from {path} -> {", ".join(written)}')
        datasets.append((label, joints))
        summary[label] = dict(stats, source=os.path.abspath(path))

    if len(datasets) > 1 and not args.no_comparison:
        if len(datasets) > len(dp.SERIES):
            print(f'Skipping the comparison figure: {len(datasets)} inputs exceed the '
                  f'{len(dp.SERIES)} distinguishable series; compare in smaller groups.')
        else:
            written = plot_comparison(
                datasets, thresholds, args.bin_width, args.log, args.clip_max,
                os.path.join(args.output_dir, 'joint_count_distribution_comparison'),
                args.pdf)
            print(f'[comparison] {len(datasets)} datasets -> {", ".join(written)}')

    if args.pooled:
        pooled = np.concatenate([j for _, j in datasets])
        written = plot_dataset(
            'all', pooled, summarize(pooled, thresholds), thresholds, args.bin_width,
            args.log, args.clip_max,
            os.path.join(args.output_dir, 'joint_count_distribution_pooled'), args.pdf,
            title='Distribution of Joint Counts across Skeletons')
        print(f'[pooled] {pooled.size:,} skeletons -> {", ".join(written)}')

    summary_path = os.path.join(args.output_dir, 'joint_count_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary -> {summary_path}\n')
    print_table(summary, thresholds)


if __name__ == '__main__':
    main()
