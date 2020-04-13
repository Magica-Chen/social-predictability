#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_functions.py
# (c) Zexun Chen, 2020-04-13
# sxtpy2010@gmail.com

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def hist_entropy(user_stats, l=12, w=6, n_bins=100, mode='talk'):
    """ Histogram plot for entropy and more
    :param user_stats: dataset, containing all ego-alter pair info
    :param l: int, long
    :param w: int, width
    :param n_bins: int, how many bins shown in the plot
    :param mode: string, see from seaborn, available,'talk', 'notebook',
    'paper', 'poster'.
    :return: None
    """
    LZentropy = user_stats[user_stats['Included Rank'] == 1]['LZ_entropy'].dropna()
    CrossEntropy = user_stats['CE_alter'].dropna()
    CrossEntropyEgo = user_stats['CCE_ego_alter'].dropna()
    CumCrossEntropy = user_stats[user_stats['Included Rank'] == 1]['CCE_alters'].dropna()
    CumCrossEntropyEgo = user_stats[user_stats['Included Rank'] == 1]['CCE_ego_alters'].dropna()

    fig, ax = plt.subplots(figsize=(l, w))
    sns.set_context(mode)
    sns.distplot(LZentropy, label='LZ Entropy', bins=n_bins)
    sns.distplot(CrossEntropy, label='Cross Entropy: alter only', bins=n_bins)
    sns.distplot(CrossEntropyEgo, label='Cumulative Cross Entropy: ego + alter', bins=n_bins)
    sns.distplot(CumCrossEntropy, label='Cumulative Cross Entropy: alters only', bins=n_bins)
    sns.distplot(CumCrossEntropyEgo, label='Cumulative Cross Entropy: ego + alters', bins=n_bins)
    # plt.title('Entropy and Cross entropy')
    ax.set(xlabel='Entropy (bits)', ylabel='Density')
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
    plt.show()


def hist_pred(user_stats, l=12, w=6, n_bins=100, mode='talk'):
    """ Histogram plot for predictability and theirs
    :param user_stats: dataset, containing all ego-alter pair info
    :param l: int, long
    :param w: int, width
    :param n_bins: int, how many bins shown in the plot
    :param mode: string, see from seaborn, available,'talk', 'notebook',
    'paper', 'poster'.
    :return: None
    """
    pred = user_stats[user_stats['Included Rank'] == 1]['Pi'].dropna()
    pred_alter = user_stats['Pi_alter'].dropna()
    pred_alter_ego = user_stats['Pi_ego_alter'].dropna()
    pred_alters = user_stats[user_stats['Included Rank'] == 1]['Pi_alters'].dropna()
    pred_alters_ego = user_stats[user_stats['Included Rank'] == 1]['Pi_ego_alters'].dropna()

    fig, ax = plt.subplots(figsize=(l, w))
    sns.set_context(mode)
    sns.distplot(pred, label=r'$\Pi$: ego', bins=n_bins)
    sns.distplot(pred_alter, label=r'$\Pi$: alter only', bins=n_bins)
    sns.distplot(pred_alter_ego, label=r'$\Pi$: ego + alter', bins=n_bins)
    sns.distplot(pred_alters, label=r'$\Pi$: alters only', bins=n_bins)
    sns.distplot(pred_alters_ego, label=r'$\Pi$: ego + alters', bins=n_bins)
    # plt.title('Entropy and Cross entropy')
    ax.set(xlabel='Predictability $\Pi$', ylabel='Density')
    ax.legend()
    plt.show()


def paper_hist(user_stats, l=15, w=6, n_bins=100, mode='talk',
               figsave=False, format='eps'):
    """ Histogram plot for entropy and predictability given by the James' paper
    :param user_stats: dataset, containing all ego-alter pair info
    :param l: int, long
    :param w: int, width
    :param n_bins: int, how many bins shown in the plot
    :param mode: string, see from seaborn, available,'talk', 'notebook',
    'paper', 'poster'.
    :param figsave: whether the figure will be saved
    :param format: png, eps, pdf, and so on.
    :return: None
    """
    LZentropy = user_stats[user_stats['Included Rank'] == 1]['LZ_entropy'].dropna()
    pred = user_stats[user_stats['Included Rank'] == 1]['Pi'].dropna()
    # only include the most frequent alter
    CrossEntropy = user_stats[user_stats['Included Rank'] == 1]['CE_alter'].dropna()
    pred_alter = user_stats[user_stats['Included Rank'] == 1]['Pi_alter'].dropna()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(l, w))
    sns.set_context(mode)
    sns.distplot(LZentropy, label='Entropy', bins=n_bins, ax=ax1, kde=False)
    sns.distplot(CrossEntropy, label='Cross Entropy', bins=n_bins, ax=ax1, kde=False)
    ax1.set(xlabel='Entropy (bits)', ylabel='Count')
    ax1.legend(loc='best')

    sns.distplot(pred, label='Entropy', bins=n_bins, ax=ax2, kde=False)
    sns.distplot(pred_alter, label='Cross Entropy', bins=n_bins, ax=ax2, kde=False)
    ax2.set(xlabel='Predictability', ylabel='Counts')
    ax2.legend(loc='best')
    plt.tight_layout()
    plt.show()

    if figsave:
        title = 'Meetupers of Alter vs Pi.' + format
        fig.savefig(title, bbox_inches='tight')


def paper_interaction(user_stats, aspect='alter', threshold=None, interval=None,
                      l=12, w=8, n_bins=200, mode='talk',
                      figsave=False, format='eps'):
    """ Number of meetupers of ego vs predictability
    :param aspect: str, from the view of ego or alter
    :param user_stats: dataset, containing all ego-alter pair info
    :param l: int, long
    :param w: int, width
    :param n_bins: int, how many bins shown in the plot
    :param mode: string, see from seaborn, available,'talk', 'notebook',
    'paper', 'poster'.
    :param threshold: int, the largest number of alters included
    :param interval: int, the interval shown in axis
    :param figsave: whether the figure will be saved
    :param format: png, eps, pdf, and so on.

    :return: None
    """
    if aspect == 'alter':
        iterim = user_stats
        iterim = iterim.rename(columns={'n_alter_meetupers': 'count'})
    else:
        N_meetupers = user_stats.groupby('userid_x')['userid_y'].count().reset_index(name='count')
        iterim = user_stats.merge(N_meetupers, left_on='userid_y', right_on='userid_x', how='left')

    if threshold is None:
        threshold = max(iterim['count'])

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True,
                                   figsize=(l, w))
    sns.set_context(mode)
    sns.pointplot(x="count", y="Pi_alter", data=iterim[iterim['count'] <= threshold],
                  ci=95, join=False, ax=ax1)
    if interval is None:
        interval = round(threshold / 20)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    ax1.set(ylabel='Predictability', xlabel='')
    ax1.set_rasterized(True)

    sns.distplot(iterim[iterim['count'] <= threshold]['count'], bins=n_bins, kde=False, ax=ax2)
    x_name = 'Number of meetupers of '+ aspect
    ax2.set(xlabel=x_name, ylabel='Count')

    fig.tight_layout()
    plt.show()

    if figsave:
        title = 'Meetupers of Alter vs Pi.' + format
        fig.savefig(title, bbox_inches='tight')


def num_point_plot(user_stats, name, threshold=None, interval=None, l=15, w=6, mode='talk',
                   control=False, figsave=False, format='eps'):
    """ number of included alters vs entropy or predictability
    :param user_stats: result dataset
    :param name: string, currently only accept 'entropy' or 'predictability'
    :param threshold: int, the largest number of alters included
    :param interval: int, the interval shown in axis
    :param l: int, long
    :param w: int, width
    :param mode: string, see from seaborn, available,'talk', 'notebook',
    'paper', 'poster'.
    :param control: whether add temporal control and social control in the plot
    :param figsave: whether the figure will be saved
    :param format: png, eps, pdf, and so on.

    :return: None
    """
    fig, ax = plt.subplots(figsize=(l, w))
    sns.set_context(mode)

    if name is 'entropy':
        if control:
            CCE = pd.melt(user_stats, id_vars=['Included Rank'],
                          value_vars=['CCE_ego_alters', 'CCE_alters',
                                      'CCE_ego_alters_tr', 'CCE_alters_tr',
                                      'CCE_ego_alters_sr', 'CCE_alters_sr'],
                          var_name='CCE')
            baseline = user_stats[user_stats['Included Rank'] == 1]['LZ_entropy'].mean()
            CCE_legend = ['Ego only', 'Alters + ego', 'Alters only',
                          'Alters + ego (TC)', 'Alters only (TC)',
                          'Alters + ego (SC)', 'Alters only (SC)']
            if threshold is None:
                threshold = len(set(user_stats['Included Rank'].tolist()))
        else:
            CCE = pd.melt(user_stats, id_vars=['Included Rank'],
                          value_vars=['CCE_ego_alters', 'CCE_alters'],
                          var_name='CCE')
            baseline = user_stats[user_stats['Included Rank'] == 1]['LZ_entropy'].mean()
            CCE_legend = ['Ego only', 'Alters + ego', 'Alters only']
            if threshold is None:
                threshold = len(set(user_stats['Included Rank'].tolist()))

    elif name is 'predictability':
        if control:
            CCE = pd.melt(user_stats, id_vars=['Included Rank'],
                          value_vars=['Pi_ego_alters', 'Pi_alters',
                                      'Pi_ego_alters_tr', 'Pi_alters_tr',
                                      'Pi_ego_alters_sr', 'Pi_alters_sr'
                                      ],
                          var_name='CCE')
            baseline = user_stats[user_stats['Included Rank'] == 1]['Pi'].mean()
            CCE_legend = ['Ego only', 'Alters + ego', 'Alters only',
                          'Alters + ego (TC)', 'Alters only (TC)',
                          'Alters + ego (SC)', 'Alters only (SC)']
            if threshold is None:
                threshold = len(set(user_stats['Included Rank'].tolist()))

        else:
            CCE = pd.melt(user_stats, id_vars=['Included Rank'],
                          value_vars=['Pi_ego_alters', 'Pi_alters'],
                          var_name='CCE')
            baseline = user_stats[user_stats['Included Rank'] == 1]['Pi'].mean()
            CCE_legend = ['Ego only', 'Alters + ego', 'Alters only']
            if threshold is None:
                threshold = len(set(user_stats['Included Rank'].tolist()))

    else:
        raise ValueError('Only available for entropy and predictability')

    sns.pointplot(x="Included Rank", y="value", hue='CCE',
                  data=CCE[CCE['Included Rank'] <= threshold],
                  ci=95, join=False, ax=ax)
    ax.axhline(y=baseline, color='black', linestyle='--', label='Ego')
    leg_handles = ax.get_legend_handles_labels()[0]

    if interval is None:
        interval = round(threshold / 20)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    if name is 'predictability':
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
        ax.legend(leg_handles, CCE_legend)
        ax.set(xlabel='Number of included alters', ylabel='Predictability')
    elif name is 'entropy':
        ax.legend(leg_handles, CCE_legend, loc='upper left', bbox_to_anchor=(1.04, 1))
        ax.set(xlabel='Number of included alters', ylabel='Cumulative Cross Entropy')

    plt.show()

    if figsave:
        title = name + '-point.' + format
        fig.savefig(title, bbox_inches='tight')