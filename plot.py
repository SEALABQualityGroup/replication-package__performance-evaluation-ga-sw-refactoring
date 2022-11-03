import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sys import stderr
from os.path import basename

initial_solutions = {
    'train-ticket' : [0, 0, 15, -0.6579247],
    'cocome' :       [0, 0, 12, -0.5026818],
}

class Experiment:

    def __init__(self, pattern):
        self.pattern = pattern
        self.runs = 0
        self.data = self._read_data()
        self._get_problem_info()

    def _read_data(self):
        dfs = []
        for f in glob(self.pattern):
            df = pd.read_csv(f)
            df['run'] = re.search(r'_(\d+)_run', f).groups()[0]
            df['iteration'] = df.index
            df['execution_time(ms)'] = df['execution_time(ms)'] / 10**3
            df.rename(columns={'execution_time(ms)':'execution_time(sec)'},
                      inplace=True)
            df['total_memory_after(B)'] = df['total_memory_after(B)'] / 1024**3
            df.rename(columns={'total_memory_after(B)':'total_memory_after(GiB)'},
                      inplace=True)
            dfs.append(df)
            self.runs += 1
        return pd.concat(dfs)

    def _get_problem_info(self):
        self.algorithm = self.data['algorithm'].iloc[0]
        tag = self.data['problem_tag'].iloc[0]
        m = re.match(r'(?P<casestudy>[^_]+)_.+_MaxEval_(?P<eval>\d+)', tag)
        if m is None:
            print('Unable to parse the problem tag: {}'.format(tag), file=stderr)
            return
        d = m.groupdict()
        self.casestudy = d['casestudy']
        self.eval = d['eval']

class Pareto:

    def __init__(self, filename):
        self.filename = filename
        self._parse_filename()
        self.data = self._read_data()

    def _parse_filename(self):
        m = re.search(r'(?P<casestudy>[^_]+)__(?P<algorithm>[^_]+)'
                      r'_(?P<eval>\d+)_eval__reference_pareto.csv',
                      basename(self.filename))
        if m is None:
            print('Unable to parse: {}'.format(self.filename), file=stderr)
            return
        d = m.groupdict()
        for k, v in d.items():
            setattr(self, k, v)

    def _read_data(self):
        df = pd.read_csv(self.filename)
        df['perfQ'] = -df['perfQ']
        df['reliability'] = -df['reliability']
        df['pas'] = df['pas'].astype(int)
        df.rename(columns={'pas':'#PAs'}, inplace=True)
        df['algorithm'] = self.algorithm.upper()
        return df


def read_runs_data(data_dir):
    patterns = set([re.sub(r'_\d_run', '_*_run', f)
                    for f in glob('{}/*.csv'.format(data_dir))])
    return [Experiment(p) for p in patterns]

def read_reference_data(data_dir):
    return [Pareto(f) for f in glob('{}/*.csv'.format(data_dir))]

def get_metric_name(metric):
    return metric.replace('_', ' ')\
            .replace('(', ' (')\
            .replace(' after', '')\
            .replace('total ', '')

def plot_setup():
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    return fig, ax

def plot_save(fig, filename):
    fig.tight_layout()
    fig.savefig(filename)
    fig.clear()
    plt.close(fig)
    print('Saved to:', filename)

def cmp_algo(exps, metric):
    eval = '102'
    casestudies = set([e.casestudy for e in exps])
    for cs in casestudies:
        df = pd.concat([e.data for e in exps
                        if e.casestudy == cs and e.eval == eval])
        df = pd.melt(df, id_vars=['algorithm', 'run', 'iteration'],
                     value_vars=[metric])
        df.sort_values('algorithm', inplace=True)
        metric_name = metric.replace('(', '_').replace(')', '_')
        filename = 'figs/cmp_algo_{}_{}_{}_eval.pdf'.format(
                cs, metric_name, eval)
        lineplot(df, metric, 'algorithm', filename)

def lineplot(df, metric, hue, filename):
    fig, ax = plot_setup()

    sns.lineplot(data=df, x='iteration', y='value', hue='algorithm',
                 palette='cubehelix', ax=ax)
    ax.set_ylabel(get_metric_name(metric))
    ax.legend(ncols=3, loc='upper left')

    plot_save(fig, filename)

def plot_reference(exps):
    eval = '102'
    casestudies = set([e.casestudy for e in exps])
    for cs in casestudies:
        data = pd.concat([e.data for e in exps if e.casestudy == cs])
        data.sort_values('algorithm', inplace=True)
        init_sol = pd.DataFrame([initial_solutions[cs]],
                    columns=['perfQ', '#changes', '#PAs', 'reliability'])
        filename = 'figs/reference_{}_{}_eval.pdf'.format(cs, eval)
        scatter_reference(data, init_sol, filename)

def scatter_reference(data, init_sol, filename):
    fig, ax = plot_setup()

    sns.scatterplot(data=data, x='perfQ', y='reliability', hue='algorithm',
                    size='#PAs', sizes=(20, 200), style='algorithm',
                    markers=['o', 's', '^'], palette='cubehelix', ax=ax)

    ax.plot(init_sol['perfQ'], -init_sol['reliability'], color='k',
            marker='x', markersize=10, markeredgewidth=2)
    ax.text(init_sol['perfQ'] + data['perfQ'].max() * .016,
            -init_sol['reliability'] - data['reliability'].max() * .02,
            'initial', color='k')
    ax.set_xlim([-.15, .45])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower left')

    plot_save(fig, filename)


if __name__ == "__main__":
    RUNS_DIR = 'data/runs'
    exps = read_runs_data(RUNS_DIR)
    cmp_algo(exps, 'execution_time(sec)')
    cmp_algo(exps, 'total_memory_after(GiB)')

    REFERENCE_DIR = 'data/reference'
    exps = read_reference_data(REFERENCE_DIR)
    plot_reference(exps)
