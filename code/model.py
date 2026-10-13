"""Load the pinned, locally patched upstream implementation."""
from vendor.networks import IMVFullLSTM, IMVTensorLSTM


def build_model(kind, input_dim, units):
    if kind not in ('full', 'tensor'):
        raise ValueError(kind)
    return {'full': IMVFullLSTM, 'tensor': IMVTensorLSTM}[kind](input_dim, 1, units)
