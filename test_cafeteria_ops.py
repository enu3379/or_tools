"""검증기가 실제로 위반을 잡는지까지 확인한다. 실행: pytest -q"""
import copy

import numpy as np
import pytest

from cafeteria_ops import Sim, baseline, make_instance, solve_exact, solve_flow, solve_lpfix, validate


@pytest.fixture(scope="module")
def inst():
    return make_instance(n_rest=3, n_emp=300, n_build=3)


@pytest.fixture(scope="module")
def plan(inst):
    return solve_exact(inst, 60, 0.02)


def test_exact_is_valid(inst, plan):
    assert validate(inst, plan) == []


def test_lpfix_is_valid(inst):
    assert validate(inst, solve_lpfix(inst, 60, 0.02)) == []


def test_flow_is_valid(inst):
    assert validate(inst, solve_flow(inst)) == []


def test_special_day_opens_special_line():
    i = make_instance(n_rest=4, n_emp=300, n_build=3, special=True)
    p = solve_exact(i, 60, 0.05)
    assert validate(i, p) == []
    k = i.lines.index("special")
    assert all(p.x[r, k] == 1 for r in i.special_sites)


@pytest.mark.parametrize("attack, word", [
    (lambda p: p.h.__imul__(0), "처리 능력"),
    (lambda p: p.q.__imul__(0), "준비량"),
    (lambda p: p.e.__imul__(0), "시프트"),
    (lambda p: p.n.__iadd__(50), "수요"),
])
def test_validator_catches_violation(inst, plan, attack, word):
    broken = copy.deepcopy(plan)
    attack(broken)
    assert any(word in m for m in validate(inst, broken))


def test_same_seed_gives_same_scenarios(inst):
    s = Sim(inst, baseline(inst))
    assert s.run(5, seed=1).equals(s.run(5, seed=1))
    assert np.isfinite(s.run(5).values).all()
