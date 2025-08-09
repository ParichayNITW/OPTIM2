
import streamlit as st
import pandas as pd
from math import pi

from dra_utils import get_ppm_for_dr, get_dr_for_ppm
from pipeline_model import solve_pipeline

st.set_page_config(page_title="Pipeline Optima — Daily Scheduler", layout="wide")

# --- session defaults
if "stations" not in st.session_state: st.session_state["stations"] = []
if "FLOW" not in st.session_state: st.session_state["FLOW"] = 1000.0
if "op_mode" not in st.session_state: st.session_state["op_mode"] = "Flow rate"
if "terminal_name" not in st.session_state: st.session_state["terminal_name"] = "Terminal"
if "terminal_elev" not in st.session_state: st.session_state["terminal_elev"] = 0.0
if "terminal_head" not in st.session_state: st.session_state["terminal_head"] = 25.0  # enforce 25 m
if "RateDRA" not in st.session_state: st.session_state["RateDRA"] = 300.0
if "Price_HSD" not in st.session_state: st.session_state["Price_HSD"] = 90.0

# --- Sidebar global
with st.sidebar:
    st.header("Global Inputs")
    st.session_state["FLOW"] = st.number_input("Flow (m³/h)", min_value=0.0, value=st.session_state["FLOW"], step=100.0)
    st.session_state["RateDRA"] = st.number_input("DRA Cost (INR/kg)", min_value=0.0, value=st.session_state["RateDRA"], step=10.0)
    st.session_state["Price_HSD"] = st.number_input("Power Cost (INR/kWh)", min_value=0.0, value=st.session_state["Price_HSD"], step=1.0)
    st.session_state["terminal_head"] = st.number_input("Min Residual Head at Peak (m)", min_value=0.0, value=st.session_state["terminal_head"], step=1.0)

    st.markdown("### Operating Mode")
    mode = st.radio("Choose input mode", ["Flow rate", "Pumping Schedule"], horizontal=False, key="op_mode")

    st.markdown("**Linefill at 07:00 Hrs (Volumetric)**")
    if "linefill_vol_df" not in st.session_state:
        st.session_state["linefill_vol_df"] = pd.DataFrame({
            "Product": ["Product-1","Product-2","Product-3"],
            "Volume (m³)": [50000.0, 40000.0, 15000.0],
            "Viscosity (cSt)": [5.0, 12.0, 15.0],
            "Density (kg/m³)": [810.0, 825.0, 865.0]
        })
    st.session_state["linefill_vol_df"] = st.data_editor(
        st.session_state["linefill_vol_df"],
        num_rows="dynamic", key="linefill_vol_editor", use_container_width=True
    )
    if mode == "Pumping Schedule":
        st.markdown("**Pumping Plan for the Day (Order of Pumping)**")
        if "day_plan_df" not in st.session_state:
            st.session_state["day_plan_df"] = pd.DataFrame({
                "Product": ["Product-4","Product-5","Product-6","Product-7"],
                "Volume (m³)": [12000.0, 6000.0, 10000.0, 8000.0],
                "Viscosity (cSt)": [3.0, 10.0, 15.0, 4.0],
                "Density (kg/m³)": [800.0, 840.0, 880.0, 770.0]
            })
        st.session_state["day_plan_df"] = st.data_editor(
            st.session_state["day_plan_df"],
            num_rows="dynamic", key="day_plan_editor", use_container_width=True
        )

st.title("Pipeline Optima — 4‑Hourly Optimizer (07:00 → 03:00)")

# Dummy stations editor (minimal)
st.subheader("Stations (minimal demo fields)")
if not st.session_state["stations"]:
    st.session_state["stations"] = [
        {"name":"Origin","L": 20.0, "D":0.711, "t":0.007, "eps":1e-5, "max_pumps":2, "min_speed_rpm":1500, "max_speed_rpm":3000, "dr_step":5, "max_dr":40, "eff":0.75, "power_rate_inr_per_kwh": st.session_state["Price_HSD"], "kv_ref":5.0},
        {"name":"Mid","L": 40.0, "D":0.711, "t":0.007, "eps":1e-5, "max_pumps":2, "min_speed_rpm":1500, "max_speed_rpm":3000, "dr_step":5, "max_dr":40, "eff":0.75, "power_rate_inr_per_kwh": st.session_state["Price_HSD"], "kv_ref":12.0},
        {"name":"TermPS","L": 40.0, "D":0.711, "t":0.007, "eps":1e-5, "max_pumps":2, "min_speed_rpm":1500, "max_speed_rpm":3000, "dr_step":5, "max_dr":40, "eff":0.75, "power_rate_inr_per_kwh": st.session_state["Price_HSD"], "kv_ref":15.0},
    ]

st.json(st.session_state["stations"])

def pipe_cross_section_area_m2(stations):
    D = float(stations[0].get("D",0.711)); t=float(stations[0].get("t",0.007))
    d=max(D-2*t,1e-3); return float(pi*d*d/4.0)

def map_vol_linefill_to_segments(vol_table: pd.DataFrame, stations: list[dict]):
    A = pipe_cross_section_area_m2(stations)
    batches=[]
    for _,r in vol_table.iterrows():
        V=float(r["Volume (m³)"]); if V<=0: continue
        length_km=(V/A)/1000.0
        batches.append({"len_km": length_km, "kv": float(r["Viscosity (cSt)"]), "rho": float(r["Density (kg/m³)"])})
    seg_kv=[]; seg_rho=[]
    seg_lengths=[s.get("L",0.0) for s in stations]
    i=0; rem=batches[0]["len_km"] if batches else 0.0; kv=batches[0]["kv"] if batches else 0.0; rho=batches[0]["rho"] if batches else 0.0
    for L in seg_lengths:
        need=L
        while need>1e-9:
            if rem<=1e-9:
                i+=1
                if i>=len(batches): rem=need
                else:
                    rem=batches[i]["len_km"]; kv=batches[i]["kv"]; rho=batches[i]["rho"]
            take=min(need, rem)
            if need==L:
                seg_kv.append(kv); seg_rho.append(rho)
            need-=take; rem-=take
    return seg_kv, seg_rho

def shift_vol_linefill(vol_table: pd.DataFrame, pumped_m3: float, day_plan: pd.DataFrame|None):
    vol_table=vol_table.copy()
    vol_table["Volume (m³)"]=vol_table["Volume (m³)"].astype(float)
    remaining=pumped_m3
    i=0
    while remaining>1e-9 and i<len(vol_table):
        v=vol_table.at[i,"Volume (m³)"]
        take=min(v, remaining)
        v_new=v-take
        vol_table.at[i,"Volume (m³)"]=v_new
        remaining-=take
        if v_new<=1e-9: i+=1
    vol_table=vol_table.iloc[i:].reset_index(drop=True)
    if day_plan is not None:
        for _,r in day_plan.iterrows():
            vol_table = pd.concat([vol_table, pd.DataFrame([{
                "Product": r.get("Product",""),
                "Volume (m³)": float(r.get("Volume (m³)",0.0)),
                "Viscosity (cSt)": float(r.get("Viscosity (cSt)",0.0)),
                "Density (kg/m³)": float(r.get("Density (kg/m³)",0.0)),
            }])], ignore_index=True)
    return vol_table

# DRA band state across time (plug flow)
def advance_dra_bands(bands, v_list, dt_hours, seg_lengths_km):
    """bands: list of dicts per station: [{'pos_km':0.0, 'ppm':X}, ...]
       Advance by dx = v*dt along cumulative length; returns per-segment %DR profile array.
       For simplicity, only most-recent band determines DR in a segment.
    """
    dt = dt_hours*3600.0
    seg_dr = [0.0]*len(seg_lengths_km)
    cum = [0.0]
    for L in seg_lengths_km: cum.append(cum[-1]+L)
    # per-station overlay (take latest band)
    for st_i, st_bands in enumerate(bands):
        for b in st_bands:
            b["pos_km"] += (v_list[st_i]*dt)/1000.0  # km
        # compute DR from latest ppm at current pos
        latest = st_bands[-1] if st_bands else None
        if latest is None: continue
        # Determine which segments are covered from station start to pos
        pos = latest["pos_km"]
        for seg_i, (a,bound) in enumerate(zip(cum[:-1], cum[1:])):
            if pos > a:
                cover = min(pos, bound) - a
                if cover>0:
                    # Use viscosity of that segment to convert ppm->%DR
                    # We'll fetch kv after mapping; here we simply store ppm; conversion in app later
                    seg_dr[seg_i] = max(seg_dr[seg_i], latest["ppm"])
    return seg_dr

st.markdown("---")
st.subheader("Run 4‑hourly Schedule")

run_day = st.button("🕒 Run Schedule (07:00→03:00, every 4h)")

if run_day:
    import copy
    stations = copy.deepcopy(st.session_state["stations"])
    term = {"name": st.session_state["terminal_name"], "elev": st.session_state["terminal_elev"], "min_residual": st.session_state["terminal_head"]}
    vol_df = st.session_state["linefill_vol_df"]
    plan_df = st.session_state.get("day_plan_df") if st.session_state["op_mode"]=="Pumping Schedule" else None

    # Determine FLOW for schedule
    if st.session_state["op_mode"]=="Pumping Schedule":
        daily_m3 = float(plan_df["Volume (m³)"].astype(float).sum()) if len(plan_df) else 0.0
        FLOW = daily_m3/24.0
    else:
        FLOW = st.session_state["FLOW"]

    hours = [7,11,15,19,23,3]  # 6 runs
    interval_h = 4.0
    A = pipe_cross_section_area_m2(stations)
    seg_lengths = [s.get("L",0.0) for s in stations]

    # Build initial kv/rho per segment
    current_vol = vol_df.copy()
    kv_list, rho_list = map_vol_linefill_to_segments(current_vol, stations)

    # Bands per station
    dra_bands = [[] for _ in stations]  # each band: {'ppm': x, 'pos_km': 0}

    # Helper to convert ppm profile to %DR profile using segment viscosities
    from dra_utils import get_dr_for_ppm
    def ppm_to_dr_profile(ppm_profile, kv_list):
        return [get_dr_for_ppm(kv, ppm) if ppm>0 else 0.0 for kv,ppm in zip(kv_list, ppm_profile)]

    rows=[]; linefill_snapshots=[]
    for ti, hr in enumerate(hours):
        # Compute velocities per segment
        Q = FLOW/3600.0
        v_list = []
        for s in stations:
            D=float(s.get("D",0.711)); t=float(s.get("t",0.007))
            Aseg=pi*max(D-2*t,1e-3)**2/4.0
            v_list.append(Q/max(Aseg,1e-9))

        # Advance bands from previous step
        if ti>0:
            ppm_profile = advance_dra_bands(dra_bands, v_list, interval_h, seg_lengths)
            dr_profile = ppm_to_dr_profile(ppm_profile, kv_list)
        else:
            dr_profile = [0.0]*len(stations)  # no benefit at t0

        # Optimize this 4h window (free PPM & pumps) using current DR profile
        res = solve_pipeline(
            stations, term, FLOW, kv_list, rho_list,
            st.session_state["RateDRA"], st.session_state["Price_HSD"],
            context=current_vol.to_dict(), dr_profile=dr_profile, interval_hours=interval_h, min_peak_head_m=st.session_state["terminal_head"]
        )
        if res.get("error"):
            st.error(f"Optimization failed at {hr:02d}:00: {res.get('message','')}"); st.stop()

        # Record Summary-like fields
        row={"Time": f"{hr:02d}:00", "Total Cost (4h)": float(res["total_cost"])*(interval_h/24.0)}
        if "power_fuel_cost_day" in res: row["Power+Fuel (4h)"]=float(res["power_fuel_cost_day"])*(interval_h/24.0)
        if "dra_cost_day" in res: row["DRA (4h)"]=float(res["dra_cost_day"])*(interval_h/24.0)
        for stn in stations:
            k = stn["name"].lower().replace(" ","_")
            for suf,label in [
                ("num_pumps","Num Pumps"),("speed","Speed (RPM)"),("dra_ppm","DRA (PPM)"),
                ("dra_dr","DRA (%DR)"),("power_kw","Power (kW)")
            ]:
                key=f"{suf}_{k}"
                if key in res: row[f"{stn['name']} - {label}"]=res[key]
        rows.append(row)

        # Create new 07+4h decision band per station with chosen PPM (takes effect from next step)
        for i,stn in enumerate(stations):
            k = stn["name"].lower().replace(" ","_")
            ppm = float(res.get(f"dra_ppm_{k}", 0.0))
            dra_bands[i].append({"ppm": ppm, "pos_km": 0.0})

        # Save linefill snapshot
        linefill_snapshots.append({"time": f"{hr:02d}:00", "table": current_vol.copy()})

        # Shift linefill for next step; append day plan only at first shift
        if ti < len(hours)-1:
            pumped = FLOW*interval_h
            current_vol = shift_vol_linefill(current_vol, pumped, plan_df if (ti==0 and plan_df is not None) else None)
            kv_list, rho_list = map_vol_linefill_to_segments(current_vol, stations)

    df = pd.DataFrame(rows).fillna("")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download 4‑hourly Results", df.to_csv(index=False), file_name="schedule_4h_results.csv")

    # Export linefill snapshots as CSV-in-zip
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for snap in linefill_snapshots:
            csv = snap["table"].to_csv(index=False)
            zf.writestr(f"linefill_{snap['time'].replace(':','')}.csv", csv)
    st.download_button("Download Linefill Snapshots (zip)", buf.getvalue(), file_name="linefill_07_to_03_zip.zip")
