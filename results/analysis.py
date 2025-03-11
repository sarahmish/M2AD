"""
Generate the figures and results for the paper: "M2AD: Multi-Sensor Multi-System
Anomaly Detection through Global Scoring and Calibrated Thresholding."
"""
import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

_COLORS = ["#3CAEA3", "#20639B", "#173F5F", "#ED553B", "#F6D55C"]
_PALETTE = sns.color_palette(_COLORS)

OUTPUT_PATH = Path('output')
os.makedirs(OUTPUT_PATH, exist_ok=True)

def _savefig(fig, name, figdir=OUTPUT_PATH):
    figdir = Path(figdir)
    for ext in ['.png', '.pdf', '.eps']:
        fig.savefig(figdir.joinpath(name+ext), bbox_inches='tight')


def _scores(df, iteration=False):
    if iteration:
        scores = df.groupby(['dataset', 'pipeline', 'iteration'])[['tp', 'fp', 'fn']].sum()
    else:
        scores = df.groupby(['dataset', 'pipeline'])[['tp', 'fp', 'fn']].sum()
    
    scores = scores.eval('precision = tp / (tp + fp)')
    scores = scores.eval('recall = tp / (tp + fn)')
    scores = scores.eval('f1 = 2 * precision * recall / (precision + recall)')
    scores = scores.eval('f05 = (1 + 0.5**2) * precision * recall / ((0.5**2 * precision) + recall)')
    
    return scores.reset_index()


def table2(df):
    scores = _scores(df, iteration=True)
    return scores.groupby(['pipeline', 'dataset'])[['f1', 'f05']].agg(['mean', 'std']).unstack()


def figure3(df):
    scores = _scores(df)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(5.5, 2.5), sharey=True)
    sns.barplot(scores, x='dataset', y='f1', hue='pipeline', palette=_COLORS, linewidth=0.5, edgecolor='k', ax=ax1)
    sns.barplot(scores, x='dataset', y='f05', hue='pipeline', palette=_COLORS, linewidth=0.5, edgecolor='k', ax=ax2, legend=False)
    
    plt.ylim(-0.02, 1.05)
    ax1.set_ylabel('$F1$ / $F_{0.5}$')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    
    handles, labels = ax1.get_legend_handles_labels()
    ax1.get_legend().remove()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=2)
    
    ax1.set_title('$F1$ Score')
    ax2.set_title('$F_{0.5}$ Score')

    _savefig(fig, 'figure3')


def figure4a(df):
    fig, ax = plt.subplots(ncols=1, figsize=(3.5, 3), sharey=True)
    
    sns.barplot(df, x='#', y='value', hue='method', palette=_COLORS, linewidth=0.5, edgecolor='k', width=.7, ax=ax)
    plt.title('Top Contributing Sensors')
    plt.ylabel('Accuracy')
    plt.xlabel('')
    plt.ylim(-0.02, 1.15)
    plt.legend(loc='upper center', ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

    _savefig(fig, 'figure4a')