import argparse
from pathlib import Path

from .block1 import run_s0_synthetic, run_s1_exchange_rate, run_s2_query_kernel
from .plots import save_block1_plots


PRESETS = {
    'paper': {
        's0': {},
        's1': {},
        's2': {},
    },
    'quick': {
        's0': dict(
            n_models=4,
            d_act=2,
            d_obs=4,
            m_totals=(24,),
            alphas=(0.0, 0.5, 1.0),
            coverage_modes=('oracle', 'pca'),
            n_seeds=2,
        ),
        's1': dict(
            n_models=4,
            d_act=2,
            d_obs=4,
            epsilons=(0.25, 0.5, 1.0),
            m_p_values=(2, 4),
            m_u_grid=(2, 4, 8),
            coverage_modes=('oracle', 'pca'),
            n_seeds=2,
        ),
        's2': dict(
            n_models=4,
            d_act=2,
            d_obs=4,
            n_queries=24,
            d_seps=(0.0, 1.0, 2.0),
            alphas=(0.0,),
            coverage_modes=('oracle', 'pca'),
            n_seeds=2,
        ),
    },
}


def _save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_block1_experiments(experiment, preset):
    config = PRESETS[preset]
    results = {}

    if experiment in {'s0', 'all'}:
        results['s0'] = run_s0_synthetic(**config['s0'])

    if experiment in {'s1', 'all'}:
        s1_summary, s1_search = run_s1_exchange_rate(return_search=True, **config['s1'])
        results['s1_summary'] = s1_summary
        results['s1_search'] = s1_search

    if experiment in {'s2', 'all'}:
        results['s2'] = run_s2_query_kernel(**config['s2'])

    return results


def save_block1_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 's0' in results:
        _save_csv(results['s0'], output_dir / 's0_results.csv')
    if 's1_summary' in results:
        _save_csv(results['s1_summary'], output_dir / 's1_summary.csv')
    if 's1_search' in results:
        _save_csv(results['s1_search'], output_dir / 's1_search.csv')
    if 's2' in results:
        _save_csv(results['s2'], output_dir / 's2_results.csv')

    plot_dir = output_dir / 'plots'
    save_block1_plots(
        plot_dir,
        s0_df=results.get('s0'),
        s1_summary_df=results.get('s1_summary'),
        s2_df=results.get('s2'),
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Run Block I unpaired DKPS synthetic experiments.')
    parser.add_argument('--experiment', choices=['s0', 's1', 's2', 'all'], default='all')
    parser.add_argument('--preset', choices=sorted(PRESETS), default='paper')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f'experiments/unpaired/results/{args.preset}')
    results = run_block1_experiments(args.experiment, args.preset)
    save_block1_results(results, output_dir)

    print(f'wrote outputs to {output_dir}')
    for name in sorted(results):
        print(f'  {name}: {len(results[name])} rows')


if __name__ == '__main__':
    main()
