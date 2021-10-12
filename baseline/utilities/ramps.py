import numpy as np


def exp_rampup(current, rampup_length):
    """Exponential rampup inspired by https://arxiv.org/abs/1610.02242
        Args:
            current: float, current step of the rampup
            rampup_length: float: length of the rampup

    """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def exp_rampdown(current, rampdown_length):
    """Exponential rampup inspired by https://arxiv.org/abs/1610.02242
        Args:
            current: float, current step of the rampup
            rampup_length: float: length of the rampup

    """
    rampdown_length = current - rampdown_length
    if rampdown_length > 0:
        aaa = 1.000005 ** rampdown_length
        return aaa
    else:
        return 1
