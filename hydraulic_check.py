import numpy as np
import pandas as pd
import streamlit as st

from pipeline_model import _segment_hydraulics, head_to_kgcm2

# Pump curve data at DOL speed (2970 RPM)
PUMP_CURVE = {
    "A": {
        "flow": np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1687], dtype=float),
        "head": np.array([430, 423, 417, 407, 395, 380, 355, 325, 290, 245], dtype=float),
        "range": (563, 1687),
    },
    "B": {
        "flow": np.array([7.9, 374.8, 569.4, 818.2, 1094.5, 1318.2, 1486.1, 1614.1, 1700, 1800, 1900], dtype=float),
        "head": np.array([342, 327, 324, 318, 316, 299, 289, 279, 270, 260, 240], dtype=float),
        "range": (616, 1900),
    },
}

DOL_SPEED = 2970.0
SUCTION_HEAD = 60.0  # m
START_ELEV = 3.6
PEAK_ELEV = 267.0
TERMINAL_ELEV = 46.0
PEAK_MIN = 15.0
TERMINAL_MIN = 50.0
MOP_KGCM2 = 60.0
L1 = 370.0
L2 = 130.0
DIAM_INNER = 0.762 - 2 * 0.00792
ROUGHNESS = 45e-6


def _interp_head(pump_type: str, flow: float) -> float:
    data = PUMP_CURVE[pump_type]
    flows = data["flow"]
    heads = data["head"]
    if flow <= flows[0]:
        return heads[0]
    if flow >= flows[-1]:
        return heads[-1]
    return float(np.interp(flow, flows, heads))


def _single_pump_head(pump_type: str, flow: float, rpm: float) -> float:
    q_equiv = flow * DOL_SPEED / rpm if rpm > 0 else flow
    head_dol = _interp_head(pump_type, q_equiv)
    return head_dol * (rpm / DOL_SPEED) ** 2


def _group_head(flow: float, pump_type: str, rpm: float, count: int, arrangement: str) -> float:
    if count <= 0:
        return 0.0
    if arrangement == "parallel" and count > 1:
        per_flow = flow / count
        return _single_pump_head(pump_type, per_flow, rpm)
    return _single_pump_head(pump_type, flow, rpm) * count


def _combo_head(flow: float, rpm: float, combo: dict) -> float:
    head = 0.0
    head += _group_head(flow, "A", rpm, combo.get("A", 0), combo.get("arrA", "series"))
    head += _group_head(flow, "B", rpm, combo.get("B", 0), combo.get("arrB", "series"))
    return head


def _flow_limits(combo: dict) -> tuple[float, float]:
    ranges = []
    for ptype in ["A", "B"]:
        cnt = combo.get(ptype, 0)
        if cnt <= 0:
            continue
        min_f, max_f = PUMP_CURVE[ptype]["range"]
        arr = combo.get(f"arr{ptype}", "series")
        if arr == "parallel" and cnt > 1:
            ranges.append((cnt * min_f, cnt * max_f))
        else:
            ranges.append((min_f, max_f))
    if not ranges:
        return (0.0, 0.0)
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


def _evaluate(combo: dict, rpm: float, rho: float, kv: float, dra: float):
    qmin, qmax = _flow_limits(combo)
    if qmax <= qmin:
        return None
    flows = np.linspace(qmin, qmax, 50)
    best = None
    for q in flows:
        hp = _combo_head(q, rpm, combo)
        sdh = SUCTION_HEAD + hp
        hl1, *_ = _segment_hydraulics(q, L1, DIAM_INNER, ROUGHNESS, kv, dra)
        hl2, *_ = _segment_hydraulics(q, L2, DIAM_INNER, ROUGHNESS, kv, dra)
        peak_head = sdh - hl1 - (PEAK_ELEV - START_ELEV)
        term_head = peak_head - hl2 - (TERMINAL_ELEV - PEAK_ELEV)
        if peak_head >= PEAK_MIN and term_head >= TERMINAL_MIN:
            press = head_to_kgcm2(sdh, rho)
            if press <= MOP_KGCM2:
                best = (q, sdh, peak_head, term_head)
    if best is None:
        return None
    q, sdh, peak_head, term_head = best
    return {
        "Combination": combo["name"],
        "RPM": rpm,
        "Flow (m3/hr)": round(q, 1),
        "Discharge Head (m)": round(sdh, 1),
        "Head at Peak (m)": round(peak_head, 1),
        "Terminal Head (m)": round(term_head, 1),
    }


def _combo_label(a: int, arrA: str, b: int, arrB: str) -> str:
    parts = []
    if a > 0:
        parts.append(f"{a}A" + (" parallel" if a > 1 and arrA == "parallel" else "" if a == 1 else f" {arrA}"))
    if b > 0:
        parts.append(f"{b}B" + (" parallel" if b > 1 and arrB == "parallel" else "" if b == 1 else f" {arrB}"))
    return " + ".join(parts)


def _generate_combos():
    combos = []
    for a in range(0, 5):
        for b in range(0, 3):
            if a == 0 and b == 0:
                continue
            arrA_list = ["series", "parallel"] if a > 1 else ["series"]
            arrB_list = ["series", "parallel"] if b > 1 else ["series"]
            for arrA in arrA_list:
                for arrB in arrB_list:
                    combos.append({
                        "A": a,
                        "B": b,
                        "arrA": arrA,
                        "arrB": arrB,
                        "name": _combo_label(a, arrA, b, arrB),
                    })
    return combos


def analyze_all_combinations(visc: float, rho: float, dra: float) -> pd.DataFrame:
    rpm_list = list(range(2000, int(DOL_SPEED) + 1, 100))
    records = []
    for combo in _generate_combos():
        best = None
        for rpm in rpm_list:
            res = _evaluate(combo, rpm, rho, visc, dra)
            if res and (best is None or res["Flow (m3/hr)"] > best["Flow (m3/hr)"]):
                best = res
        if best:
            records.append(best)
    return pd.DataFrame(records)


def hydraulic_app():
    st.title("Hydraulic Feasibility Check")
    visc = st.number_input("Viscosity (cSt)", value=5.0, step=0.1)
    rho = st.number_input("Density (kg/m³)", value=820.0, step=1.0)
    dra = st.slider("Drag Reduction (%)", 0, 35, 0)
    if st.button("Run hydraulic analysis"):
        df = analyze_all_combinations(visc, rho, dra)
        if df.empty:
            st.warning("No feasible pump combinations found.")
        else:
            st.dataframe(df)

__all__ = ["hydraulic_app", "analyze_all_combinations"]
