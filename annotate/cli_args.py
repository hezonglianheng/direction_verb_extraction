# encoding: utf8
"""Lightweight annotation selection rules shared by CLI execution and preflight."""

import argparse


class ConsistentSelection(argparse.Action):
    """Allow an identical repeated selection, never silently choose the last conflict."""

    def __call__(self, parser, namespace, values, option_string=None):
        previous = getattr(namespace, self.dest, None)
        if previous is not None and previous != values:
            raise argparse.ArgumentError(self, f"重复且冲突：{previous} / {values}")
        setattr(namespace, self.dest, values)


def resolve_graph_selection(graph, engine, *, engine_option="--engine"):
    """Resolve compatible old/new selectors without importing graphs or model code."""
    if engine == "all-in-one":
        if graph not in (None, "all_in_one"):
            raise ValueError(f"{engine_option} all-in-one 只能与 --graph all_in_one 组合")
        return "all_in_one"
    if engine == "staged" and graph == "all_in_one":
        raise ValueError(f"{engine_option} staged 与 --graph all_in_one 冲突；"
                         f"请去掉 {engine_option} 或改为 all-in-one")
    return graph or "pair"
