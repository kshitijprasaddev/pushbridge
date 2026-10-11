"""A platform-rounding allowance must not hide changed reset conditions."""
import numpy as np
import pytest

from scripts.audit_rl_followup import verify_declared_reset
from pushbridge.evaluation import make_trials


def test_reset_regeneration_accepts_one_ulp_but_rejects_changed_conditions():
    trial = make_trials(1, 11939)[0]
    row = {'friction': trial['friction'], 'goal': trial['goal'].copy(),
           'initial_puck': trial['puck'].copy(), 'initial_tool': trial['tool'].copy()}
    row['initial_tool'][0] = np.nextafter(row['initial_tool'][0], np.inf)
    assert verify_declared_reset(row, trial) < 1e-14
    row['initial_tool'][0] += 1e-10
    with pytest.raises(AssertionError):
        verify_declared_reset(row, trial)
    row['initial_tool'] = trial['tool'].copy()
    row['initial_tool'][0] = np.nan
    with pytest.raises(AssertionError):
        verify_declared_reset(row, trial)
    row['initial_tool'] = trial['tool'][0]
    with pytest.raises(AssertionError):
        verify_declared_reset(row, trial)
