#!/usr/bin/env python3
"""식당 운영 최적화: 데이터 -> 현행안 -> 최적화 -> 검증 -> 시뮬레이션 -> 차트.

필수: numpy, pandas, scipy, matplotlib
선택: ortools (없으면 --mode flow 가 scipy.optimize.linprog 로 같은 네트워크를 푼다)

실행 예:
  python cafeteria_ops.py --restaurants 3 --employees 300 --time-limit 30        # 동작 확인
  python cafeteria_ops.py --restaurants 30 --employees 5000 --time-limit 300     # 전체 규모
  python cafeteria_ops.py --restaurants 30 --employees 5000 --special            # 특식일
  python cafeteria_ops.py --mode lpfix ...   # exact 가 느릴 때
  python cafeteria_ops.py --mode flow ...    # 그래도 느릴 때
  python cafeteria_ops.py --data-dir data    # 실제 CSV 사용 (--export-data 로 형식 확인)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.special import softmax

BASE_LINES = ["korean", "western", "snack", "asian"]
SPECIAL = "special"


# ---------------------------------------------------------------------------
# 1. 입력 데이터
# ---------------------------------------------------------------------------
@dataclass
class Inst:
    groups: pd.DataFrame       # building, start, end, desired, taste(line 번호), d
    seats: np.ndarray          # 식당별 좌석 수
    walk: np.ndarray           # 건물 x 식당 도보 시간(분)
    lines: list[str]
    slot_min: int = 15
    n_slots: int = 12          # 11:00~14:00
    max_lines: int = 3         # 식당당 동시 운영 라인 수
    mu_per_hour: float = 60.0  # 배식 인원 1명의 시간당 처리량
    dine_min: int = 30         # 식사 시간
    max_staff: int = 6         # 라인당 슬롯별 배식 인원 상한
    shift_len: int = 8         # 시프트 길이(슬롯), 15분 x 8 = 2시간
    shift_starts: tuple = (0, 2, 4)
    nearest: int = 5           # 그룹별 후보 식당 수
    special_sites: tuple = ()  # 특식 라인을 반드시 여는 식당
    special_cap: float = 1e9   # 특식 재료 한도(인분)
    # 비용(원). 모두 가정값이므로 실제 값으로 바꿀 것
    time_value: float = 200.0     # 근무자 시간 가치(원/분)
    pref_penalty: float = 1500.0  # 선호하지 않는 메뉴
    unserved_penalty: float = 15000.0
    labor_per_shift: float = 30000.0
    line_fixed: float = 50000.0
    open_per_slot: float = 10000.0
    food_unit: float = 3000.0
    capacity_buffer: float = 0.85  # 계획 때 처리량의 85%까지만 사용(변동 대비)

    @property
    def R(self): return len(self.seats)
    @property
    def K(self): return len(self.lines)
    @property
    def T(self): return self.n_slots
    @property
    def G(self): return len(self.groups)
    @property
    def mu(self): return self.mu_per_hour * self.slot_min / 60.0
    @property
    def D(self): return max(1, round(self.dine_min / self.slot_min))
    @property
    def shift_cover(self):
        return np.array([[s <= t < s + self.shift_len for s in self.shift_starts]
                         for t in range(self.T)], dtype=int)  # [t, s]

    def cost(self, g, r, k, t):
        row = self.groups.iloc[g]
        walk = self.walk[int(row.building), r] * self.time_value
        pref = 0.0 if k == int(row.taste) else self.pref_penalty
        late = abs(t - row.desired) * self.slot_min * self.time_value * 0.5
        return walk + pref + late

    def candidates(self, g):
        b = int(self.groups.iloc[g].building)
        return np.argsort(self.walk[b])[: self.nearest]


def make_instance(n_rest=30, n_emp=5000, special=False, seed=7, n_build=12, slot_min=15) -> Inst:
    """합성 데이터. 실제 데이터가 있으면 load_instance 를 쓴다."""
    rng = np.random.default_rng(seed)
    bxy = rng.uniform(0, 1500, (n_build, 2))
    rxy = rng.uniform(0, 1500, (n_rest, 2))
    walk = np.linalg.norm(bxy[:, None] - rxy[None], axis=2) / 70.0 + 1  # 70m/분
    n_slots = int(180 / slot_min)
    per_hour = 60 // slot_min
    windows = [(0, per_hour), (per_hour // 2, per_hour * 3 // 2), (per_hour, per_hour * 2),
               (per_hour * 2, per_hour * 3 - 1)]
    win_share = [0.2, 0.35, 0.3, 0.15]
    lines = BASE_LINES + ([SPECIAL] if special else [])
    taste = np.array([0.4, 0.25, 0.15, 0.2])
    if special:
        taste = np.r_[taste * 0.7, 0.3]
    bshare = rng.dirichlet(np.full(n_build, 4.0))
    total = int(n_emp * (1.1 if special else 1.0))  # 특식일 이용자 10% 증가 가정
    rows = []
    for b in range(n_build):
        for w, (s, e) in enumerate(windows):
            for k in range(len(lines)):
                d = total * bshare[b] * win_share[w] * taste[k]
                rows.append(dict(building=b, start=s, end=e, desired=(s + e - 1) / 2,
                                 taste=k, d=round(d, 2)))
    groups = pd.DataFrame(rows)
    groups = groups[groups.d >= 1].reset_index(drop=True)
    seats = rng.integers(120, 420, n_rest).astype(float)
    sites = tuple(int(i) for i in np.argsort(-seats)[:max(1, n_rest // 8)]) if special else ()
    return Inst(groups, seats, walk, lines, slot_min=slot_min, n_slots=n_slots,
                shift_len=int(120 / slot_min),
                shift_starts=tuple(range(0, max(1, n_slots - int(120 / slot_min)) + 1,
                                         max(1, n_slots // 6))),
                special_sites=sites, special_cap=total * 0.35)


def export_instance(i: Inst, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    i.groups.to_csv(d / "groups.csv", index=False)
    pd.DataFrame({"seats": i.seats}).to_csv(d / "restaurants.csv", index=False)
    pd.DataFrame(i.walk).to_csv(d / "walk.csv", index=False)
    json.dump({"lines": i.lines, "special_sites": list(i.special_sites),
               "special_cap": i.special_cap}, open(d / "meta.json", "w"))


def load_instance(d: Path) -> Inst:
    """groups.csv(building,start,end,desired,taste,d), restaurants.csv(seats),
    walk.csv(건물 행 x 식당 열, 분), meta.json(lines,special_sites,special_cap)"""
    m = json.load(open(d / "meta.json"))
    return Inst(pd.read_csv(d / "groups.csv"), pd.read_csv(d / "restaurants.csv").seats.to_numpy(float),
                pd.read_csv(d / "walk.csv").to_numpy(float), m["lines"],
                special_sites=tuple(m["special_sites"]), special_cap=m["special_cap"])


def arcs(i: Inst):
    """가능한 (그룹, 식당, 라인, 슬롯) 배정 조합. 간선 축소가 여기서 일어난다."""
    out = []
    for g in range(i.G):
        row = i.groups.iloc[g]
        for r in i.candidates(g):
            for k in range(i.K):
                if i.lines[k] == SPECIAL and int(r) not in i.special_sites:
                    continue
                for t in range(int(row.start), int(row.end)):
                    out.append((g, int(r), k, t))
    return out


# ---------------------------------------------------------------------------
# 2. 운영안 자료구조
# ---------------------------------------------------------------------------
@dataclass
class Plan:
    name: str
    x: np.ndarray            # [r,k] 라인 개설
    y: np.ndarray            # [r,t] 운영
    h: np.ndarray            # [r,k,t] 배식 인원
    e: np.ndarray            # [r,s] 시프트별 고용
    q: np.ndarray            # [r,k] 준비량
    f: dict = field(default_factory=dict)  # (g,r,k,t) -> 배정 인원
    v: np.ndarray | None = None            # [r,k,t] 배식량
    n: np.ndarray | None = None            # [g] 미이용
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3. MILP
# ---------------------------------------------------------------------------
class Builder:
    def __init__(self):
        self.n = 0; self.c = []; self.lb = []; self.ub = []; self.int = []
        self.ri = []; self.ci = []; self.val = []; self.rl = []; self.ru = []

    def var(self, cost=0.0, lb=0.0, ub=np.inf, integer=False):
        self.c.append(cost); self.lb.append(lb); self.ub.append(ub); self.int.append(int(integer))
        self.n += 1
        return self.n - 1

    def con(self, terms, lo, hi):
        row = len(self.rl)
        for idx, co in terms:
            self.ri.append(row); self.ci.append(idx); self.val.append(co)
        self.rl.append(lo); self.ru.append(hi)

    def solve(self, time_limit, gap, relax=False):
        A = sp.csr_matrix((self.val, (self.ri, self.ci)), shape=(len(self.rl), self.n))
        return milp(np.array(self.c), constraints=LinearConstraint(A, self.rl, self.ru),
                    bounds=Bounds(self.lb, self.ub),
                    integrality=np.zeros(self.n) if relax else np.array(self.int),
                    options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})


def build_model(i: Inst, labor_mult=1.0, fix_x=None, fix_y=None):
    b = Builder()
    inf = np.inf
    A = arcs(i)
    R, K, T, S = i.R, i.K, i.T, len(i.shift_starts)
    wait_cost = i.time_value * i.slot_min
    V = {"f": {}, "n": {}, "x": {}, "y": {}, "h": {}, "v": {}, "L": {}, "q": {}, "e": {}, "z": {}}
    for a in A:
        V["f"][a] = b.var(i.cost(*a) + 0)
    for g in range(i.G):
        V["n"][g] = b.var(i.unserved_penalty)
    for r in range(R):
        for k in range(K):
            allowed = i.lines[k] != SPECIAL or r in i.special_sites
            lo = 1.0 if i.lines[k] == SPECIAL and r in i.special_sites else 0.0
            hi = 1.0 if allowed else 0.0
            if fix_x is not None:
                lo = hi = float(fix_x[r, k])
            V["x"][r, k] = b.var(i.line_fixed, lo, hi, True)
            V["q"][r, k] = b.var(i.food_unit)
            for t in range(T):
                V["h"][r, k, t] = b.var(0, 0, i.max_staff, True)
                V["v"][r, k, t] = b.var(0)
                V["L"][r, k, t] = b.var(wait_cost)
        for t in range(T):
            lo, hi = 0.0, 1.0
            if fix_y is not None:
                lo = hi = float(fix_y[r, t])
            V["y"][r, t] = b.var(i.open_per_slot, lo, hi, True)
            V["z"][r, t] = b.var(0, 0, 1, True)
        for s in range(S):
            V["e"][r, s] = b.var(i.labor_per_shift * labor_mult, 0, inf, True)

    by_g, by_rkt = {}, {}
    for a, idx in V["f"].items():
        by_g.setdefault(a[0], []).append(idx)
        by_rkt.setdefault(a[1:], []).append(idx)

    for g in range(i.G):                                      # 수요
        d = float(i.groups.d.iloc[g])
        b.con([(idx, 1) for idx in by_g.get(g, [])] + [(V["n"][g], 1)], d, d)
    cover = i.shift_cover
    mu = i.mu * i.capacity_buffer
    for r in range(R):
        for k in range(K):
            for t in range(T):
                terms = [(V["L"][r, k, t], 1), (V["v"][r, k, t], 1)]      # 대기 흐름 보존
                terms += [(idx, -1) for idx in by_rkt.get((r, k, t), [])]
                if t > 0:
                    terms.append((V["L"][r, k, t - 1], -1))
                b.con(terms, 0, 0)
                b.con([(V["v"][r, k, t], 1), (V["h"][r, k, t], -mu)], -inf, 0)      # 배식량
                b.con([(V["h"][r, k, t], 1), (V["x"][r, k], -i.max_staff)], -inf, 0)  # 라인 개설
                b.con([(V["h"][r, k, t], 1), (V["y"][r, t], -i.max_staff)], -inf, 0)  # 운영 여부
            b.con([(V["L"][r, k, T - 1], 1)], 0, 0)                  # 종료 시 대기 0
            b.con([(V["v"][r, k, t], 1) for t in range(T)] + [(V["q"][r, k], -1)], -inf, 0)  # 준비량
            cap = i.special_cap if i.lines[k] == SPECIAL else 5000.0
            b.con([(V["q"][r, k], 1), (V["x"][r, k], -cap)], -inf, 0)
        extra = sum(i.lines[k] == SPECIAL and r in i.special_sites for k in range(K))
        b.con([(V["x"][r, k], 1) for k in range(K)], -inf, i.max_lines + extra)  # 라인 수
        for t in range(T):
            terms = [(V["h"][r, k, t], 1) for k in range(K)]         # 인력
            terms += [(V["e"][r, s], -int(cover[t, s])) for s in range(S)]
            b.con(terms, -inf, 0)
            terms = [(V["v"][r, k, tau], 1) for k in range(K)         # 좌석
                     for tau in range(max(0, t - i.D + 1), t + 1)]
            b.con(terms, -inf, float(i.seats[r]))
            terms = [(V["y"][r, t], 1), (V["z"][r, t], -1)]           # 연속 운영
            if t > 0:
                terms.append((V["y"][r, t - 1], -1))
            b.con(terms, -inf, 0)
        b.con([(V["z"][r, t], 1) for t in range(T)], -inf, 1)
    sk = [k for k in range(K) if i.lines[k] == SPECIAL]
    if sk:                                                            # 특식 재료 한도
        b.con([(V["q"][r, sk[0]], 1) for r in range(R)], -inf, i.special_cap)
    return b, V


def extract(i: Inst, V, sol, name, meta) -> Plan:
    R, K, T, S = i.R, i.K, i.T, len(i.shift_starts)
    x = np.zeros((R, K)); y = np.zeros((R, T)); h = np.zeros((R, K, T)); v = np.zeros((R, K, T))
    q = np.zeros((R, K)); e = np.zeros((R, S)); n = np.zeros(i.G)
    for (r, k), idx in V["x"].items(): x[r, k] = round(sol[idx])
    for (r, k), idx in V["q"].items(): q[r, k] = sol[idx]
    for (r, t), idx in V["y"].items(): y[r, t] = round(sol[idx])
    for (r, k, t), idx in V["h"].items(): h[r, k, t] = round(sol[idx])
    for (r, k, t), idx in V["v"].items(): v[r, k, t] = sol[idx]
    for (r, s), idx in V["e"].items(): e[r, s] = round(sol[idx])
    for g, idx in V["n"].items(): n[g] = sol[idx]
    f = {a: sol[idx] for a, idx in V["f"].items() if sol[idx] > 1e-7}
    return Plan(name, x, y, h, e, q, f, v, n, meta)


def solve_exact(i: Inst, time_limit, gap, labor_mult=1.0, name="optimized") -> Plan:
    b, V = build_model(i, labor_mult)
    t0 = time.time()
    res = b.solve(time_limit, gap)
    if res.x is None:
        raise RuntimeError(f"시간 안에 정수해를 찾지 못함: {res.message} -> --mode lpfix 또는 flow 사용")
    meta = dict(mode="exact", seconds=round(time.time() - t0, 1), n_vars=b.n, n_int=sum(b.int),
                n_cons=len(b.rl), gap=None if res.mip_gap is None else round(float(res.mip_gap), 4),
                objective=round(float(res.fun)), status=res.status)
    return extract(i, V, res.x, name, meta)


def solve_lpfix(i: Inst, time_limit, gap, name="optimized") -> Plan:
    """LP 완화 -> 라인/운영 시간 고정 -> 인력만 정수로 다시 푼다."""
    t0 = time.time()
    b, V = build_model(i)
    lp = b.solve(time_limit, 1.0, relax=True)
    if lp.x is None:
        raise RuntimeError("LP 완화 실패")
    x = np.array([[lp.x[V["x"][r, k]] for k in range(i.K)] for r in range(i.R)])
    use = np.zeros((i.R, i.K, i.T))
    for (r, k, t), idx in V["v"].items():
        use[r, k, t] = lp.x[idx]
    fx = np.zeros_like(x)
    for r in range(i.R):
        extra = sum(i.lines[k] == SPECIAL and r in i.special_sites for k in range(i.K))
        score = use[r].sum(axis=1)
        for k in np.argsort(-score)[: i.max_lines + extra]:
            if score[k] > 1e-6:
                fx[r, k] = 1
        for k in range(i.K):
            if i.lines[k] == SPECIAL and r in i.special_sites:
                fx[r, k] = 1
    fy = np.zeros((i.R, i.T))
    for r in range(i.R):
        on = np.where(use[r].sum(axis=0) > 1e-6)[0]
        if len(on):
            fy[r, on.min(): on.max() + 1] = 1
    b2, V2 = build_model(i, fix_x=fx, fix_y=fy)
    res = b2.solve(max(10, time_limit - (time.time() - t0)), gap)
    if res.x is None:
        raise RuntimeError(f"고정 후 해를 찾지 못함: {res.message}")
    meta = dict(mode="lpfix", seconds=round(time.time() - t0, 1), n_vars=b2.n, n_int=sum(b2.int),
                n_cons=len(b2.rl), gap=None if res.mip_gap is None else round(float(res.mip_gap), 4),
                objective=round(float(res.fun)), status=res.status,
                lp_lower_bound=round(float(lp.fun)))
    return extract(i, V2, res.x, name, meta)


# ---------------------------------------------------------------------------
# 4. 대체 경로: 개설을 탐욕법으로 정하고 배정은 최소비용흐름
# ---------------------------------------------------------------------------
def solve_flow(i: Inst, name="optimized") -> Plan:
    t0 = time.time()
    R, K, T = i.R, i.K, i.T
    # 탐욕 개설: 근처 그룹의 취향 수요가 많은 라인부터
    pull = np.zeros((R, K))
    for g in range(i.G):
        row = i.groups.iloc[g]
        c = i.candidates(g)
        pull[c, int(row.taste)] += row.d / len(c)
    x = np.zeros((R, K))
    for r in range(R):
        for k in np.argsort(-pull[r])[: i.max_lines]:
            if i.lines[k] != SPECIAL:
                x[r, k] = 1
        for k in range(K):
            if i.lines[k] == SPECIAL and r in i.special_sites:
                x[r, k] = 1
    # 고정 인원: 근처 수요에 비례하되 좌석 범위 안에서
    h = np.zeros((R, K, T))
    mu = i.mu * i.capacity_buffer
    for r in range(R):
        for k in range(K):
            if x[r, k]:
                need = pull[r, k] / max(1, T * mu) * 2.0
                h[r, k, :] = min(i.max_staff, max(1, np.ceil(need)))
    y = (h.sum(axis=1) > 0).astype(float)
    cap = np.zeros((R, K, T))
    for r in range(R):
        nl = max(1, int(x[r].sum()))
        cap[r, :, :] = np.minimum(mu * h[r], np.floor(i.seats[r] / (i.D * nl)))

    # 네트워크: S는 공급으로 표현. 노드 = 그룹 + (r,k,t) + 싱크
    A = arcs(i)
    rkt = {(r, k, t): i.G + r * K * T + k * T + t for r in range(R) for k in range(K) for t in range(T)}
    sink = i.G + R * K * T
    tail, head, ucap, cost = [], [], [], []
    def add(a, b_, u, c):
        tail.append(a); head.append(b_); ucap.append(u); cost.append(c)
    n_f = 0
    fkeys = []
    total = float(i.groups.d.sum())
    for a in A:
        if x[a[1], a[2]] and cap[a[1], a[2], a[3]] > 0:
            add(a[0], rkt[a[1:]], total, i.cost(*a)); fkeys.append(a); n_f += 1
    for g in range(i.G):
        add(g, sink, total, i.unserved_penalty)
    service_start = len(tail)
    skeys = []
    for (r, k, t), node in rkt.items():
        if cap[r, k, t] > 0:
            add(node, sink, cap[r, k, t], 0); skeys.append((r, k, t))
        if t < T - 1:
            add(node, rkt[r, k, t + 1], total, i.time_value * i.slot_min)
    d = i.groups.d.to_numpy()
    supply = np.zeros(sink + 1)
    supply[: i.G] = d
    supply[sink] = -d.sum()
    n_arcs = len(tail)
    try:
        from ortools.graph.python import min_cost_flow  # 정수 입력만 받으므로 10배/반올림
        smcf = min_cost_flow.SimpleMinCostFlow()
        sup_int = np.round(supply * 10).astype(np.int64)
        sup_int[sink] -= sup_int.sum()
        for a, b_, u, c in zip(tail, head, ucap, cost):
            smcf.add_arc_with_capacity_and_unit_cost(a, b_, int(u * 10), int(round(c)))
        for node, s in enumerate(sup_int):
            smcf.set_node_supply(node, int(s))
        if smcf.solve() != smcf.OPTIMAL:
            raise RuntimeError("최소비용흐름 실패")
        flow = np.array([smcf.flow(a) / 10 for a in range(n_arcs)])
        engine = "ortools.SimpleMinCostFlow"
    except ImportError:
        M = sp.csr_matrix((np.r_[np.ones(n_arcs), -np.ones(n_arcs)],
                           (np.r_[tail, head], np.r_[np.arange(n_arcs), np.arange(n_arcs)])),
                          shape=(sink + 1, n_arcs))
        res = linprog(cost, A_eq=M, b_eq=supply, bounds=list(zip([0] * n_arcs, ucap)), method="highs")
        if res.x is None:
            raise RuntimeError(res.message)
        flow = res.x
        engine = "scipy.linprog"
    f = {a: flow[j] for j, a in enumerate(fkeys) if flow[j] > 1e-7}
    n = flow[n_f: n_f + i.G]
    v = np.zeros((R, K, T))
    pos = service_start
    for (r, k, t) in rkt:
        if cap[r, k, t] > 0:
            v[r, k, t] = flow[pos]; pos += 1
        if t < T - 1:
            pos += 1
    e = cover_shifts(i, h)
    q = v.sum(axis=2)
    meta = dict(mode="flow", engine=engine, seconds=round(time.time() - t0, 1), n_vars=n_arcs, n_int=0,
                n_cons=sink + 1, gap=None, objective=None, status=0)
    return Plan(name, x, y, h, e, q, f, v, n, meta)


def cover_shifts(i: Inst, h: np.ndarray) -> np.ndarray:
    """슬롯별 필요 인원을 덮는 시프트 수를 작은 정수계획으로 계산."""
    S = len(i.shift_starts)
    e = np.zeros((i.R, S))
    cover = i.shift_cover
    for r in range(i.R):
        need = h[r].sum(axis=0)
        if need.sum() == 0:
            continue
        res = milp(np.ones(S), constraints=LinearConstraint(cover, need, np.inf),
                   integrality=np.ones(S), bounds=Bounds(0, np.inf))
        e[r] = np.round(res.x)
    return e


# ---------------------------------------------------------------------------
# 5. 현행안(비교 기준). 실제 대회에서는 제공된 현행 운영 규칙으로 교체한다.
# ---------------------------------------------------------------------------
def baseline(i: Inst, capacity_ratio=1.5) -> Plan:
    """모든 식당 전 시간 운영, 고정 라인, 시간대와 무관한 고정 인원."""
    R, K, T = i.R, i.K, i.T
    x = np.zeros((R, K))
    base_k = [k for k in range(K) if i.lines[k] != SPECIAL]
    for r in range(R):
        for j in range(i.max_lines):
            x[r, base_k[(r + j) % len(base_k)]] = 1
        for k in range(K):
            if i.lines[k] == SPECIAL and r in i.special_sites:
                x[r, k] = 1
    total = float(i.groups.d.sum())
    per_slot = total * capacity_ratio / (T * i.mu)  # 전체 슬롯당 필요 인원
    share = i.seats / i.seats.sum()
    h = np.zeros((R, K, T))
    for r in range(R):
        idx = np.where(x[r] > 0)[0]
        h[r, idx, :] = min(i.max_staff, max(1, int(np.ceil(per_slot * share[r] / len(idx)))))
    y = np.ones((R, T))
    return Plan("baseline", x, y, h, cover_shifts(i, h), np.zeros((R, K)), meta=dict(mode="baseline"))


# ---------------------------------------------------------------------------
# 6. 검증기: 솔버 변수를 믿지 않고 f, v, h 에서 다시 계산한다
# ---------------------------------------------------------------------------
def validate(i: Inst, p: Plan, tol=1e-4) -> list[str]:
    bad = []
    if p.v is None:
        return bad
    R, K, T = i.R, i.K, i.T
    arr = np.zeros((R, K, T))
    served_g = np.zeros(i.G)
    for (g, r, k, t), val in p.f.items():
        row = i.groups.iloc[g]
        if not int(row.start) <= t < int(row.end):
            bad.append(f"식사 가능 시간 위반 g={g} t={t}")
        arr[r, k, t] += val
        served_g[g] += val
    miss = np.abs(served_g + p.n - i.groups.d.to_numpy())
    if miss.max() > max(tol, 0.2):
        bad.append(f"수요 불일치 최대 {miss.max():.3f}")
    mu = i.mu
    if (p.v > mu * p.h + tol).any():
        bad.append("배식량 > 처리 능력")
    L = np.zeros((R, K))
    for t in range(T):
        L = L + arr[:, :, t] - p.v[:, :, t]
        if (L < -max(tol, 0.2)).any():
            bad.append(f"도착보다 많이 배식 t={t}")
    if np.abs(L).max() > max(tol, 0.2):
        bad.append(f"끼니 종료 후 대기 잔존 {np.abs(L).max():.2f}")
    if (p.h.sum(axis=2) * (1 - p.x) > 0).any():
        bad.append("미개설 라인에 인력 배치")
    if (p.h.sum(axis=1) * (1 - p.y) > 0).any():
        bad.append("미운영 시간에 인력 배치")
    limit = i.max_lines + np.array([sum(i.lines[k] == SPECIAL and r in i.special_sites
                                        for k in range(K)) for r in range(R)])
    if (p.x.sum(axis=1) > limit).any():
        bad.append("동시 운영 라인 수 초과")
    if (p.h.sum(axis=1).T > (i.shift_cover @ p.e.T) + tol).any():
        bad.append("시프트 인력 부족")
    for t in range(T):
        lo = max(0, t - i.D + 1)
        if (p.v[:, :, lo: t + 1].sum(axis=(1, 2)) > i.seats + 1e-3).any():
            bad.append(f"좌석 초과 t={t}")
    if (p.v.sum(axis=2) > p.q + 1e-3).any():
        bad.append("준비량 초과 배식")
    for r in range(R):
        on = np.where(p.y[r] > 0)[0]
        if len(on) and len(on) != on.max() - on.min() + 1:
            bad.append(f"운영 시간 불연속 r={r}")
    for k in range(K):
        if i.lines[k] == SPECIAL:
            if any(p.x[r, k] < 1 for r in i.special_sites):
                bad.append("특식 지정 식당 미개설")
            if p.q[:, k].sum() > i.special_cap + 1e-3:
                bad.append("특식 재료 한도 초과")
    return bad


# ---------------------------------------------------------------------------
# 7. 시뮬레이션: 사람은 배정대로 움직이지 않는다 (로짓 선택 + 대기열)
# ---------------------------------------------------------------------------
class Sim:
    def __init__(self, i: Inst, p: Plan, temperature=800.0, waste_buffer=1.05):
        self.i, self.p, self.temp = i, p, temperature
        A = [a for a in arcs(i) if p.h[a[1], a[2], a[3]] > 0]
        self.g = np.array([a[0] for a in A], dtype=int)
        self.rkt = np.array([np.ravel_multi_index(a[1:], (i.R, i.K, i.T)) for a in A], dtype=int)
        self.base = np.array([i.cost(*a) for a in A])
        self.wait = np.zeros((i.R, i.K, i.T))
        self.idx_by_g = [np.where(self.g == g)[0] for g in range(i.G)]
        self.d = i.groups.d.to_numpy()
        self.q = np.full((i.R, i.K), np.inf)
        for _ in range(6):            # 대기 예상 <-> 선택 고정점
            arr = self.expected_arrivals()
            _, L, _ = self.queue(arr, self.q)
            self.wait = 0.5 * self.wait + 0.5 * L / np.maximum(self.p.h * i.mu, 1) * i.slot_min
        arr = self.expected_arrivals()
        served, _, _ = self.queue(arr, self.q)
        # 두 안 모두 같은 규칙으로 준비량을 정한다: 기대 소비량 x (1+5%)
        self.q = served.sum(axis=2) * waste_buffer
        for k in range(i.K):
            if i.lines[k] == SPECIAL and self.q[:, k].sum() > i.special_cap:
                self.q[:, k] *= i.special_cap / self.q[:, k].sum()

    def probs(self):
        i = self.i
        w = self.wait.ravel()[self.rkt] * i.time_value
        u = -(self.base + w) / self.temp
        P = np.zeros(len(u)); out = np.zeros(i.G)
        for g, idx in enumerate(self.idx_by_g):
            pr = softmax(np.r_[u[idx], -i.unserved_penalty / self.temp])
            P[idx] = pr[:-1]; out[g] = pr[-1]
        return P, out

    def expected_arrivals(self, d=None):
        i = self.i
        P, _ = self.probs()
        d = self.d if d is None else d
        return np.bincount(self.rkt, weights=P * d[self.g], minlength=i.R * i.K * i.T).reshape(i.R, i.K, i.T)

    def queue(self, arr, q):
        i = self.i
        L = np.zeros(arr.shape); served = np.zeros(arr.shape)
        cur = np.zeros((i.R, i.K)); stock = q.copy()
        rate = self.p.h * i.mu
        for t in range(i.T):
            cur = cur + arr[:, :, t]
            s = np.minimum.reduce([cur, rate[:, :, t], stock])
            served[:, :, t] = s; cur = cur - s; stock = stock - s
            L[:, :, t] = cur
        return served, L, cur.sum()

    def run(self, n=200, seed=11, demand_mult=1.0, line_closed=None):
        i = self.i
        rng = np.random.default_rng(seed)
        P, out = self.probs()
        rows = []
        pl = self.p
        labor = float(pl.e.sum() * i.labor_per_shift)
        for _ in range(n):
            # 같은 seed면 두 운영안이 같은 수요 시나리오를 받는다
            day = rng.normal(1, 0.07)
            noise = rng.lognormal(0, 0.1, i.G)
            d = self.d * demand_mult * day * noise
            arr = np.bincount(self.rkt, weights=P * d[self.g],
                              minlength=i.R * i.K * i.T).reshape(i.R, i.K, i.T)
            served, L, left = self.queue(arr, self.q)
            total = d.sum()
            sv = served.sum()
            rows.append(dict(wait_min=L.sum() * i.slot_min / max(sv, 1),
                             unserved_pct=100 * (total - sv) / total,
                             waste_pct=100 * (self.q.sum() - sv) / max(self.q.sum(), 1),
                             labor_mwon=labor / 1e6,
                             peak_util_pct=100 * (served.sum(axis=1) / np.maximum(self.p.h.sum(axis=1) * i.mu, 1)).max()))
        return pd.DataFrame(rows)


def compare(a: pd.DataFrame, b: pd.DataFrame, n_boot=2000, seed=3):
    """같은 시나리오의 쌍 차이(b-a)에 대한 95% 부트스트랩 구간."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in a.columns:
        diff = (b[c] - a[c]).to_numpy()
        idx = rng.integers(0, len(diff), (n_boot, len(diff)))
        boots = diff[idx].mean(axis=1)
        out[c] = dict(current=round(a[c].mean(), 3), optimized=round(b[c].mean(), 3),
                      diff=round(diff.mean(), 3), ci95=[round(float(np.quantile(boots, .025)), 3),
                                                        round(float(np.quantile(boots, .975)), 3)])
    return out


# ---------------------------------------------------------------------------
# 8. 차트 (영문 라벨: 대회 PC에 한글 글꼴이 없어도 깨지지 않게)
# ---------------------------------------------------------------------------
INK, MUTED, BLUE, ORANGE = "#1f2328", "#6b7280", "#2563a8", "#c8641e"


def charts(i, base, opt, sb, so, cmp_, pareto, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    labels = [f"{11 + t * i.slot_min // 60}:{t * i.slot_min % 60:02d}" for t in range(i.T)]
    step = max(1, i.T // 6)

    arr_b = sb.expected_arrivals(); arr_o = so.expected_arrivals()
    _, Lb, _ = sb.queue(arr_b, sb.q); _, Lo, _ = so.queue(arr_o, so.q)
    wb = Lb.sum(axis=1); wo = Lo.sum(axis=1)
    order = np.argsort(-wb.sum(axis=1))
    vmax = max(wb.max(), wo.max(), 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 0.18 * i.R + 2), sharey=True, constrained_layout=True)
    for a, m, title in ((ax[0], wb, f"Current: {sb.run(50).wait_min.mean():.1f} min avg wait"),
                        (ax[1], wo, f"Optimized: {so.run(50).wait_min.mean():.1f} min avg wait")):
        im = a.imshow(m[order], aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        a.set_title(title, loc="left", fontsize=11)
        a.set_xticks(range(0, i.T, step), labels[::step])
        a.spines[:].set_visible(False)
        a.tick_params(length=0)
    ax[0].set_ylabel("Restaurant (sorted by current queue)")
    ax[0].set_yticks([])
    fig.colorbar(im, ax=ax, label="People waiting", shrink=0.8)
    fig.savefig(outdir / "1_queue_heatmap.png", dpi=160); plt.close(fig)

    rows = [("wait_min", "Avg wait (min)"), ("unserved_pct", "Unserved (%)"),
            ("labor_mwon", "Labor (M KRW)"), ("waste_pct", "Waste (%)")]
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), constrained_layout=True)
    for a, (c, t) in zip(ax, rows):
        vals = [cmp_[c]["current"], cmp_[c]["optimized"]]
        a.bar(["Current", "Optimized"], vals, color=[MUTED, BLUE], width=0.55)
        for j, val in enumerate(vals):
            a.text(j, val, f"{val:.1f}", ha="center", va="bottom")
        a.set_title(t, loc="left", fontsize=11); a.set_yticks([]); a.spines["left"].set_visible(False)
        a.tick_params(length=0); a.margins(y=0.2)
    fig.savefig(outdir / "2_kpi_compare.png", dpi=160); plt.close(fig)

    if pareto:
        pdf = pd.DataFrame(pareto).sort_values("labor_mwon")
        fig, a = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
        a.plot(pdf.labor_mwon, pdf.wait_min, color=BLUE, lw=2, marker="o", ms=8, label="Optimized plans")
        for _, r in pdf.iterrows():
            a.annotate(f"labor weight x{r.labor_mult:g}", (r.labor_mwon, r.wait_min),
                       textcoords="offset points", xytext=(8, 6), color=MUTED, fontsize=9)
        a.scatter([cmp_["labor_mwon"]["current"]], [cmp_["wait_min"]["current"]], color=ORANGE, s=80,
                  zorder=3, label="Current plan")
        a.set_xlabel("Labor cost (M KRW per meal period)"); a.set_ylabel("Average wait (min)")
        a.grid(axis="y", color="#e5e7eb"); a.legend(frameon=False); a.margins(0.15)
        a.set_title("How much wait does each won of labor buy?", loc="left", fontsize=11)
        fig.savefig(outdir / "3_pareto.png", dpi=160); plt.close(fig)

    B = int(i.groups.building.max()) + 1
    mat = np.zeros((B, i.R))
    for a_, val in zip(zip(so.g, so.rkt), so.probs()[0] * so.d[so.g]):
        r = np.unravel_index(a_[1], (i.R, i.K, i.T))[0]
        mat[int(i.groups.building.iloc[a_[0]]), r] += val
    keep = np.argsort(-mat.sum(axis=0))[: min(15, i.R)]
    fig, a = plt.subplots(figsize=(8, 4), constrained_layout=True)
    im = a.imshow(mat[:, keep], aspect="auto", cmap="Blues")
    a.set_xticks(range(len(keep)), [f"R{k}" for k in keep], fontsize=8)
    a.set_yticks(range(B), [f"B{b}" for b in range(B)], fontsize=8)
    a.set_xlabel(f"Top {len(keep)} restaurants by diners"); a.set_ylabel("Building")
    a.spines[:].set_visible(False); a.tick_params(length=0)
    a.set_title("Who eats where under the optimized plan", loc="left", fontsize=11)
    fig.colorbar(im, ax=a, label="Diners")
    fig.savefig(outdir / "4_flow_matrix.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# 9. 전체 실행
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restaurants", type=int, default=30)
    ap.add_argument("--employees", type=int, default=5000)
    ap.add_argument("--special", action="store_true")
    ap.add_argument("--mode", choices=["exact", "lpfix", "flow"], default="exact")
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--scenarios", type=int, default=200)
    ap.add_argument("--pareto", action="store_true", help="인건비 가중치를 바꿔 2회 더 풀기")
    ap.add_argument("--buffer", type=float, default=0.85, help="계획 시 처리량 사용 상한. 낮출수록 인력 여유 증가")
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--export-data", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    i = load_instance(a.data_dir) if a.data_dir else make_instance(a.restaurants, a.employees, a.special)
    i.capacity_buffer = a.buffer
    if a.export_data:
        export_instance(i, a.export_data)
    print(f"[data] 수요 {i.groups.d.sum():.0f}, 그룹 {i.G}, 식당 {i.R}, 라인 {i.K}, 슬롯 {i.T}, 배정 간선 {len(arcs(i))}")

    if a.mode == "exact":
        opt = solve_exact(i, a.time_limit, a.gap)
    elif a.mode == "lpfix":
        opt = solve_lpfix(i, a.time_limit, a.gap)
    else:
        opt = solve_flow(i)
    print("[solve]", opt.meta)
    bad = validate(i, opt)
    print("[validate]", "위반 없음" if not bad else bad)

    base = baseline(i)
    sb, so = Sim(i, base), Sim(i, opt)
    rb, ro = sb.run(a.scenarios), so.run(a.scenarios)
    cmp_ = compare(rb, ro)
    print("[compare]"); print(pd.DataFrame(cmp_).T.to_string())

    sens = {}
    for name, mult in (("demand -20%", 0.8), ("demand +20%", 1.2)):
        sens[name] = dict(current=sb.run(a.scenarios, demand_mult=mult).mean().round(2).to_dict(),
                          optimized=so.run(a.scenarios, demand_mult=mult).mean().round(2).to_dict())
    # 식당 1곳 폐쇄: 현행안에서 가장 붐비는 곳을 두 안 모두 닫는다
    closed = int(np.argmax(sb.expected_arrivals().sum(axis=(1, 2))))
    for tag, plan in (("current", base), ("optimized", opt)):
        p2 = Plan(plan.name, plan.x.copy(), plan.y.copy(), plan.h.copy(), plan.e.copy(), plan.q.copy())
        p2.h[closed] = 0; p2.x[closed] = 0; p2.y[closed] = 0
        sens.setdefault("largest site closed", {})[tag] = Sim(i, p2).run(a.scenarios).mean().round(2).to_dict()
    print("[sensitivity]"); print(json.dumps(sens, ensure_ascii=False, indent=1))

    pareto = [dict(labor_mult=1.0, labor_mwon=ro.labor_mwon.mean(), wait_min=ro.wait_min.mean(),
                   unserved_pct=ro.unserved_pct.mean())]
    if a.pareto and a.mode == "exact":
        for m in (0.3, 3.0):
            pm = solve_exact(i, a.time_limit, max(a.gap, 0.08), labor_mult=m)
            rm = Sim(i, pm).run(a.scenarios)
            pareto.append(dict(labor_mult=m, labor_mwon=rm.labor_mwon.mean(), wait_min=rm.wait_min.mean(),
                               unserved_pct=rm.unserved_pct.mean()))
            print(f"[pareto] x{m}: {pareto[-1]}  gap={pm.meta['gap']} {pm.meta['seconds']}s")
    charts(i, base, opt, sb, so, cmp_, pareto if len(pareto) > 1 else None, a.out)

    rows = [dict(restaurant=r, line=i.lines[k], slot=t, staff=int(opt.h[r, k, t]))
            for r in range(i.R) for k in range(i.K) for t in range(i.T) if opt.h[r, k, t] > 0]
    pd.DataFrame(rows).to_csv(a.out / "staff_plan.csv", index=False)
    json.dump(dict(solve=opt.meta, violations=bad, compare=cmp_, sensitivity=sens, pareto=pareto,
                   size=dict(demand=float(i.groups.d.sum()), groups=i.G, restaurants=i.R, lines=i.K,
                             slots=i.T, arcs=len(arcs(i)))),
              open(a.out / "results.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(f"[done] {a.out}/ 에 results.json, staff_plan.csv, 차트 저장")


if __name__ == "__main__":
    main()
