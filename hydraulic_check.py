import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from pipeline_model import _segment_hydraulics, head_to_kgcm2

# Hide Vega action buttons on all Altair charts
alt.renderers.set_embed_options(actions=False)

# Default pump curve data at rated speed
DEFAULT_PUMP_CURVE = {
    "A": {
        "flow": np.array([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1687], dtype=float),
        "head": np.array([430, 423, 417, 407, 395, 380, 355, 325, 290, 245], dtype=float),
        "mcsf": 563,
        "aor_max": 1687,
    },
    "B": {
        "flow": np.array([7.9, 374.8, 569.4, 818.2, 1094.5, 1318.2, 1486.1, 1614.1, 1700, 1800, 1900], dtype=float),
        "head": np.array([342, 327, 324, 318, 316, 299, 289, 279, 270, 260, 240], dtype=float),
        "mcsf": 616,
        "aor_max": 1900,
    },
}



def _interp_head(pump_type: str, flow: float, pump_curve: dict) -> float:
    data = pump_curve[pump_type]
    flows = data["flow"]
    heads = data["head"]
    if flow <= flows[0]:
        return heads[0]
    if flow >= flows[-1]:
        return heads[-1]
    return float(np.interp(flow, flows, heads))


def _single_pump_head(pump_type: str, flow: float, rpm: float, pump_curve: dict, dol_speed: dict) -> float:
    rated = dol_speed[pump_type]
    q_equiv = flow * rated / rpm if rpm > 0 else flow
    head_dol = _interp_head(pump_type, q_equiv, pump_curve)
    return head_dol * (rpm / rated) ** 2


def _group_head(flow: float, pump_type: str, rpm: float, count: int, arrangement: str, pump_curve: dict, dol_speed: dict) -> float:
    if count <= 0:
        return 0.0
    if arrangement == "parallel" and count > 1:
        per_flow = flow / count
        return _single_pump_head(pump_type, per_flow, rpm, pump_curve, dol_speed)
    return _single_pump_head(pump_type, flow, rpm, pump_curve, dol_speed) * count


def _combo_head(flow: float, rpm: float, combo: dict, pump_curve: dict, dol_speed: dict) -> float:
    head = 0.0
    head += _group_head(flow, "A", rpm, combo.get("A", 0), combo.get("arrA", "series"), pump_curve, dol_speed)
    head += _group_head(flow, "B", rpm, combo.get("B", 0), combo.get("arrB", "series"), pump_curve, dol_speed)
    return head


def _flow_limits(combo: dict, pump_curve: dict) -> tuple[float, float]:
    limits = []
    for ptype in ["A", "B"]:
        cnt = combo.get(ptype, 0)
        if cnt <= 0:
            continue
        min_f = pump_curve[ptype]["mcsf"]
        # Allow evaluation beyond the pump's published curve by extending the
        # maximum flow range. The head beyond the last point is treated as
        # constant by _interp_head, so we simply double the curve's max flow.
        max_f = pump_curve[ptype]["flow"][-1] * 2
        arr = combo.get(f"arr{ptype}", "series")
        if arr == "parallel" and cnt > 1:
            limits.append((cnt * min_f, cnt * max_f))
        else:
            limits.append((min_f, max_f))
    if not limits:
        return (0.0, 0.0)
    return max(l[0] for l in limits), min(l[1] for l in limits)


def _aor_limits(combo: dict, pump_curve: dict) -> tuple[float, float]:
    ranges = []
    for ptype in ["A", "B"]:
        cnt = combo.get(ptype, 0)
        if cnt <= 0:
            continue
        min_f = pump_curve[ptype]["mcsf"]
        max_f = pump_curve[ptype]["aor_max"]
        arr = combo.get(f"arr{ptype}", "series")
        if arr == "parallel" and cnt > 1:
            ranges.append((cnt * min_f, cnt * max_f))
        else:
            ranges.append((min_f, max_f))
    if not ranges:
        return (0.0, 0.0)
    return max(r[0] for r in ranges), min(r[1] for r in ranges)


def _evaluate(
    combo: dict,
    rpm: float,
    rho: float,
    kv: float,
    dra: float,
    pipe: dict,
    pump_curve: dict,
    dol_speed: dict,
):
    qmin, qmax = _flow_limits(combo, pump_curve)
    if qmax <= qmin:
        return None

    def compute(q: float):
        hp = _combo_head(q, rpm, combo, pump_curve, dol_speed)
        sdh = pipe["suction_head"] + hp
        hl1, *_ = _segment_hydraulics(q, pipe["peak_location"], pipe["diam_inner"], pipe["roughness"], kv, dra)
        hl_total, *_ = _segment_hydraulics(q, pipe["total_length"], pipe["diam_inner"], pipe["roughness"], kv, dra)
        peak_head = sdh - hl1 - (pipe["peak_elev"] - pipe["start_elev"])
        term_head = sdh - hl_total - (pipe["terminal_elev"] - pipe["start_elev"])
        req_peak = pipe["suction_head"] + (pipe["peak_elev"] - pipe["start_elev"]) + pipe["peak_min"] + hl1
        req_term = pipe["suction_head"] + (pipe["terminal_elev"] - pipe["start_elev"]) + pipe["terminal_min"] + hl_total
        req_head = max(req_peak, req_term)
        return sdh - req_head, sdh, peak_head, term_head, req_head

    dmin, _, _, _, _ = compute(qmin)
    if dmin < 0:
        return None
    dmax, _, _, _, _ = compute(qmax)
    if dmax > 0:
        q_oper = qmax
        diff, sdh, peak_head, term_head, req_head = compute(q_oper)
    else:
        q_low, q_high = qmin, qmax
        for _ in range(50):
            q_mid = 0.5 * (q_low + q_high)
            diff_mid, sdh_mid, peak_mid, term_mid, req_mid = compute(q_mid)
            if diff_mid > 0:
                q_low = q_mid
            else:
                q_high = q_mid
            if abs(diff_mid) < 1e-3:
                q_oper = q_mid
                diff, sdh, peak_head, term_head, req_head = diff_mid, sdh_mid, peak_mid, term_mid, req_mid
                break
        else:
            q_oper = q_mid
            diff, sdh, peak_head, term_head, req_head = diff_mid, sdh_mid, peak_mid, term_mid, req_mid

    press = head_to_kgcm2(sdh, rho)
    if (
        peak_head + 1e-6 < pipe["peak_min"]
        or term_head + 1e-6 < pipe["terminal_min"]
        or press > pipe["mop"] + 1e-6
        or diff < -0.01
    ):
        return None

    aor_min, aor_max = _aor_limits(combo, pump_curve)
    in_aor = aor_min <= q_oper <= aor_max
    return {
        "Combination": combo["name"],
        "RPM": rpm,
        "Flow (m3/hr)": round(q_oper, 1),
        "Discharge Head (m)": round(sdh, 1),
        "Head at Peak (m)": round(peak_head, 1),
        "Terminal Head (m)": round(term_head, 1),
        "Within AOR": in_aor,
    }


def _combo_label(a: int, arrA: str, b: int, arrB: str) -> str:
    parts = []
    if a > 0:
        parts.append(f"{a}A" + (" parallel" if a > 1 and arrA == "parallel" else "" if a == 1 else f" {arrA}"))
    if b > 0:
        parts.append(f"{b}B" + (" parallel" if b > 1 and arrB == "parallel" else "" if b == 1 else f" {arrB}"))
    return " + ".join(parts)


def _generate_combos(max_a: int, max_b: int):
    combos = []
    for a in range(0, max_a + 1):
        for b in range(0, max_b + 1):
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


def analyze_all_combinations(
    visc: float,
    rho: float,
    dra: float,
    pipe: dict,
    pump_curve: dict,
    dol_speed: dict,
    max_a: int = 4,
    max_b: int = 4,
    min_rpm: dict | None = None,
    return_specs: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    records = []
    combo_map = {}
    if min_rpm is None:
        min_rpm = {"A": 2000.0, "B": 2000.0}
    for combo in _generate_combos(max_a, max_b):
        min_speed = 0
        max_speed = float("inf")
        if combo.get("A", 0) > 0:
            min_speed = max(min_speed, min_rpm.get("A", 0))
            max_speed = min(max_speed, dol_speed.get("A", float("inf")))
        if combo.get("B", 0) > 0:
            min_speed = max(min_speed, min_rpm.get("B", 0))
            max_speed = min(max_speed, dol_speed.get("B", float("inf")))
        if max_speed < min_speed:
            continue
        rpm_list = list(range(int(min_speed), int(max_speed) + 1, 100))
        for rpm in rpm_list:
            res = _evaluate(combo, rpm, rho, visc, dra, pipe, pump_curve, dol_speed)
            if res:
                records.append(res)
                combo_map[combo["name"]] = combo
    df = pd.DataFrame(records)
    if return_specs:
        return df, combo_map
    return df


def _plot_curves(
    combo: dict,
    rpm: float,
    pipe: dict,
    pump_curve: dict,
    dol_speed: dict,
    kv: float,
    dra: float,
    rho: float,
    op_point: tuple[float, float] | None = None,
):
    _, qmax = _flow_limits(combo, pump_curve)
    flows = np.linspace(0, qmax, 100)
    pump_heads: list[float] = []
    sys_heads: list[float] = []
    peak_heads: list[float] = []
    term_heads: list[float] = []
    for q in flows:
        hp = _combo_head(q, rpm, combo, pump_curve, dol_speed)
        sdh = pipe["suction_head"] + hp
        hl1, *_ = _segment_hydraulics(q, pipe["peak_location"], pipe["diam_inner"], pipe["roughness"], kv, dra)
        hl_total, *_ = _segment_hydraulics(q, pipe["total_length"], pipe["diam_inner"], pipe["roughness"], kv, dra)
        peak = sdh - hl1 - (pipe["peak_elev"] - pipe["start_elev"])
        term = sdh - hl_total - (pipe["terminal_elev"] - pipe["start_elev"])
        req_peak = pipe["suction_head"] + (pipe["peak_elev"] - pipe["start_elev"]) + pipe["peak_min"] + hl1
        req_term = pipe["suction_head"] + (pipe["terminal_elev"] - pipe["start_elev"]) + pipe["terminal_min"] + hl_total
        pump_heads.append(sdh)
        sys_heads.append(max(req_peak, req_term))
        peak_heads.append(max(0.0, peak))
        term_heads.append(max(0.0, term))
        
    head_mop = pipe["mop"] * 10000.0 / rho if rho > 0 else 0.0
    max_head = max(pump_heads + sys_heads + [head_mop])
    df1 = pd.DataFrame({"Flow": flows, "Pump Discharge Head": pump_heads, "System Head Required": sys_heads})
    aor_min, aor_max = _aor_limits(combo, pump_curve)
    band_df = pd.DataFrame({
        "Flow_start": [aor_min],
        "Flow_end": [aor_max],
        "Head_start": [0],
        "Head_end": [max_head],
        "Curve": ["Within AOR"],
    })
    color_scale = alt.Scale(
        domain=["Pump Discharge Head", "System Head Required", "Within AOR", "MOP"],
        range=["#1f77b4", "#d62728", "#90ee90", "#9467bd"],
    )
    rect = alt.Chart(band_df).mark_rect(opacity=0.1).encode(
        x="Flow_start",
        x2="Flow_end",
        y="Head_start",
        y2="Head_end",
        color=alt.Color("Curve", scale=color_scale, legend=alt.Legend(title="")),
    )
    lines = (
        alt.Chart(df1.melt("Flow", var_name="Curve", value_name="Head"))
        .mark_line()
        .encode(
            x=alt.X("Flow", title="Flow (m3/h)"),
            y=alt.Y("Head", title="Head (m)", scale=alt.Scale(domain=(0, max_head))),
            color=alt.Color("Curve", scale=color_scale, legend=alt.Legend(title="")),
        )
    )
    mop_line = (
        alt.Chart(pd.DataFrame({"Head": [head_mop], "Curve": ["MOP"]}))
        .mark_rule()
        .encode(y="Head", color=alt.Color("Curve", scale=color_scale, legend=alt.Legend(title="")))
    )
    chart1 = rect + lines + mop_line
    if op_point is not None:
        op_df = pd.DataFrame({"Flow": [op_point[0]], "Head": [op_point[1]]})
        chart1 += alt.Chart(op_df).mark_point(color="black", size=60).encode(x="Flow", y="Head")
    chart1 = chart1.properties(
        height=350,
        padding={"bottom": 40},
        usermeta={"embedOptions": {"actions": False}},
    )
    st.altair_chart(chart1, use_container_width=True, theme=None)

    df2 = pd.DataFrame({"Flow": flows, "Peak Head": peak_heads, "Terminal Head": term_heads})
    data2 = df2.melt("Flow", var_name="Location", value_name="Head")
    max_head2 = max(peak_heads + term_heads + [pipe["peak_min"], pipe["terminal_min"]])
    chart2 = (
        alt.Chart(data2)
        .mark_line()
        .encode(
            x=alt.X("Flow", title="Flow (m3/h)"),
            y=alt.Y("Head", title="Head (m)", scale=alt.Scale(domain=(0, max_head2))),
            color=alt.Color("Location", legend=alt.Legend(title="")),
        )
    )
    rules = (
        alt.Chart(
            pd.DataFrame({
                "Head": [pipe["peak_min"], pipe["terminal_min"]],
                "Location": ["Peak Minimum", "Terminal Minimum"],
            })
        )
        .mark_rule(strokeDash=[4, 4])
        .encode(y="Head", color="Location")
    )
    chart2_full = (chart2 + rules).properties(
        height=350,
        padding={"bottom": 40},
        usermeta={"embedOptions": {"actions": False}},
    )
    st.altair_chart(chart2_full, use_container_width=True, theme=None)


def hydraulic_app():
    st.title("Hydraulic Feasibility Check")

    st.header("Fluid Properties")
    visc = st.number_input("Viscosity (cSt)", value=5.0, step=0.1)
    rho = st.number_input("Density (kg/m³)", value=820.0, step=1.0)
    dra = st.slider("Drag Reduction (%)", 0, 70, 0)

    st.header("Pipeline Parameters")
    suction_head = st.number_input("Suction head at origin (m)", value=60.0)
    total_length = st.number_input("Total pipeline length (km)", value=500.0)
    peak_location = st.number_input("Peak location from origin (km)", value=370.0)
    start_elev = st.number_input("Start elevation (m)", value=3.6)
    peak_elev = st.number_input("Peak elevation (m)", value=267.0)
    terminal_elev = st.number_input("Terminal elevation (m)", value=46.0)
    peak_min = st.number_input("Minimum head at peak (m)", value=15.0)
    terminal_min = st.number_input("Minimum terminal head (m)", value=50.0)
    mop = st.number_input("Maximum operating pressure (kg/cm²)", value=60.0)
    od_in = st.number_input("Pipeline Outside Diameter (in)", value=30.0, format="%.3f")
    wt_in = st.number_input("Pipeline Wall Thickness (in)", value=0.312, format="%.3f")
    od = od_in * 0.0254
    wt = wt_in * 0.0254
    diam_inner = od - 2 * wt
    roughness = st.number_input("Pipe roughness (microns)", value=45.0) * 1e-6

    st.header("Pump Parameters")
    dol_speed_A = st.number_input("Type A rated speed (RPM)", value=2970.0)
    min_rpm_A = st.number_input("Type A minimum speed (RPM)", value=2000.0)
    dol_speed_B = st.number_input("Type B rated speed (RPM)", value=2970.0)
    min_rpm_B = st.number_input("Type B minimum speed (RPM)", value=2000.0)
    max_a = st.number_input("Max number of Type-A pumps", min_value=0, max_value=4, value=4)
    max_b = st.number_input("Max number of Type-B pumps", min_value=0, max_value=4, value=2)
    rangeA_min = st.number_input("Type A MCSF Flow (m³/h)", value=563.0)
    rangeA_max = st.number_input("Type A AOR Max Flow (m³/h)", value=1687.0)
    rangeB_min = st.number_input("Type B MCSF Flow (m³/h)", value=616.0)
    rangeB_max = st.number_input("Type B AOR Max Flow (m³/h)", value=1900.0)
    st.subheader("Type A Pump Flow-Head Curve")
    curveA_df = st.data_editor(
        pd.DataFrame({"Flow": DEFAULT_PUMP_CURVE["A"]["flow"], "Head": DEFAULT_PUMP_CURVE["A"]["head"]}),
        num_rows="dynamic",
        key="curveA",
        use_container_width=True,
        column_config={
            "Flow": st.column_config.NumberColumn("Flow (m³/h)", format="%.1f"),
            "Head": st.column_config.NumberColumn("Head (m)", format="%.1f"),
        },
    )
    st.subheader("Type B Pump Flow-Head Curve")
    curveB_df = st.data_editor(
        pd.DataFrame({"Flow": DEFAULT_PUMP_CURVE["B"]["flow"], "Head": DEFAULT_PUMP_CURVE["B"]["head"]}),
        num_rows="dynamic",
        key="curveB",
        use_container_width=True,
        column_config={
            "Flow": st.column_config.NumberColumn("Flow (m³/h)", format="%.1f"),
            "Head": st.column_config.NumberColumn("Head (m)", format="%.1f"),
        },
    )

    pump_curve = {
        ptype: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        for ptype, data in DEFAULT_PUMP_CURVE.items()
    }
    pump_curve["A"]["mcsf"] = rangeA_min
    pump_curve["A"]["aor_max"] = rangeA_max
    pump_curve["B"]["mcsf"] = rangeB_min
    pump_curve["B"]["aor_max"] = rangeB_max
    pump_curve["A"]["flow"] = curveA_df["Flow"].to_numpy(dtype=float)
    pump_curve["A"]["head"] = curveA_df["Head"].to_numpy(dtype=float)
    pump_curve["B"]["flow"] = curveB_df["Flow"].to_numpy(dtype=float)
    pump_curve["B"]["head"] = curveB_df["Head"].to_numpy(dtype=float)

    dol_speed = {"A": dol_speed_A, "B": dol_speed_B}
    min_rpm = {"A": min_rpm_A, "B": min_rpm_B}

    pipe = {
        "suction_head": suction_head,
        "start_elev": start_elev,
        "peak_elev": peak_elev,
        "terminal_elev": terminal_elev,
        "peak_min": peak_min,
        "terminal_min": terminal_min,
        "mop": mop,
        "total_length": total_length,
        "peak_location": peak_location,
        "diam_inner": diam_inner,
        "roughness": roughness,
    }

    if st.button("Run hydraulic analysis"):
        df, combos = analyze_all_combinations(
            visc,
            rho,
            dra,
            pipe,
            pump_curve,
            dol_speed,
            max_a=max_a,
            max_b=max_b,
            min_rpm=min_rpm,
            return_specs=True,
        )
        if df.empty:
            st.warning("No feasible pump combinations found.")
            st.session_state.pop("hydraulic_results", None)
        else:
            st.session_state["hydraulic_results"] = {
                "df": df,
                "combos": combos,
                "pipe": pipe,
                "pump_curve": pump_curve,
                "dol_speed": dol_speed,
                "kv": visc,
                "dra": dra,
                "rho": rho,
            }
            st.session_state.pop("combo_choice", None)

    results = st.session_state.get("hydraulic_results")
    if results:
        df = results["df"]
        st.dataframe(df)
        labels = df.apply(lambda r: f"{r['Combination']} @ {r['RPM']} RPM", axis=1)
        choice = st.selectbox(
            "Select combination and speed for curves",
            labels,
            key="combo_choice",
        )
        if choice:
            row = df.loc[labels == choice].iloc[0]
            combo_name = row["Combination"]
            rpm_sel = float(row["RPM"])
            in_aor = bool(row["Within AOR"])
            if not in_aor:
                st.warning("Operating point lies outside AOR range.")
            _plot_curves(
                results["combos"][combo_name],
                rpm_sel,
                results["pipe"],
                results["pump_curve"],
                results["dol_speed"],
                results["kv"],
                results["dra"],
                results["rho"],
                op_point=(row["Flow (m3/hr)"], row["Discharge Head (m)"]),
            )


__all__ = ["hydraulic_app", "analyze_all_combinations", "DEFAULT_PUMP_CURVE"]
