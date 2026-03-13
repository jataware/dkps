#!/usr/bin/env python
"""Run all synthetic experiments sequentially."""

import sys
import os

# Ensure the synthetic directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s1_coverage_interaction import run_s1, plot_s1
from s2_query_mismatch import run_s2, plot_s2
from s4_exchange_rate import run_s4, plot_s4


def main():
    print('=' * 60)
    print('Experiment S1: Coverage Interaction')
    print('=' * 60)
    df1 = run_s1()
    plot_s1(df1)

    print()
    print('=' * 60)
    print('Experiment S2: Query Distribution Mismatch')
    print('=' * 60)
    df2 = run_s2()
    plot_s2(df2)

    print()
    print('=' * 60)
    print('Experiment S4: Exchange Rate')
    print('=' * 60)
    df4 = run_s4()
    plot_s4(df4)

    print()
    print('All experiments complete.')


if __name__ == '__main__':
    main()
