"""
Minimal nethook compatible with ROME_new
"""

import torch
from contextlib import contextmanager


class Trace:
    def __init__(self):
        self.input = None
        self.output = None


class TraceDict(dict):
    def __init__(self, model, layer_names, retain_input=False, retain_output=False):
        super().__init__()
        self.handles = []
        self.retain_input = retain_input
        self.retain_output = retain_output

        for name in layer_names:
            module = get_module(model, name)
            trace = Trace()
            self[name] = trace

            handle = module.register_forward_hook(
                lambda m, inp, out, tr=trace: self._hook(tr, inp, out)
            )
            self.handles.append(handle)

    def _hook(self, trace, inp, out):
        if self.retain_input:
            trace.input = inp
        if self.retain_output:
            trace.output = out

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()


def get_module(model, name):
    parts = name.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


def get_parameter(model, name):
    return get_module(model, name).weight.data.clone()


def set_parameter(model, name, value):
    get_module(model, name).weight.data.copy_(value)

