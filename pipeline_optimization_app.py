import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from pipeline_model import solve_pipeline

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(
    page_title="Mixed Integer Non Linear Convex Optimization of Pipeline Operations",
    layout="wide"
)



# ---------------------
# Custom CSS
# ---------------------
st.markdown(
    """
    <style>
      /* DARK MODE BASE */
      body, .block-container, .sidebar .sidebar-content {
        background-color: #0e1117 !important;
        color: #fafafa !important;
      }
      /* YOUR FROSTED-GLASS METRIC CARDS */
      .stMetric > div {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(5px);
        border-radius: 8px;
        padding: 12px;
        color: #FFFFFF;
        text-align: center;
      }
      .stMetric .metric-value,
      .stMetric .metric-label {
        display: block !important;
        width: 100% !important;
        text-align: center !important;
      }
      .section-title {
        font-size: 1.3rem; font-weight: 600;
        color: #FFFFFF; margin-top: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Mixed Integer Non Linear Convex Optimization of Pipeline Operations")


# ---------------------
# Sidebar Inputs
# ---------------------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Adjust Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)",      value=1000.0, step=10.0)
        KV        = st.number_input("Viscosity (cSt)",        value=1.0,    step=0.1)
        rho       = st.number_input("Density (kg/m³)",        value=850.0,  step=10.0)
        SFC_J     = st.number_input("SFC Jamnagar (gm/bhp/hr)", value=210.0, step=1.0)
        SFC_R     = st.number_input("SFC Rajkot (gm/bhp/hr)",   value=215.0, step=1.0)
        SFC_S     = st.number_input("SFC Surendranagar (gm/bhp/hr)", value=220.0, step=1.0)
        RateDRA   = st.number_input("DRA Rate (INR/L)",        value=1.0,    step=0.1)
        Price_HSD = st.number_input("HSD Rate (INR/L)",        value=90.0,   step=0.5)
    run = st.button("🚀 Run Optimization")

if run:
    with st.spinner("Solving pipeline optimization..."):
        res = solve_pipeline(FLOW, KV, rho, SFC_J, SFC_R, SFC_S, RateDRA, Price_HSD)

    stations = ["Vadinar","Jamnagar","Rajkot","Surendranagar","Viramgam"]
    params = {
        "Vadinar":        {"d":0.697,"L":46.7,"e":0.00004},
        "Jamnagar":      {"d":0.697,"L":67.9,"e":0.00004},
        "Rajkot":        {"d":0.697,"L":40.2,"e":0.00004},
        "Chotila":       {"d":0.697,"L":60.0,"e":0.00004},
        "Surendranagar": {"d":0.697,"L":60.0,"e":0.00004}
    }
    static_heads = {"Vadinar":16,"Jamnagar":89,"Rajkot":119,"Chotila":-152,"Surendranagar":-57}

    # KPI Cards
    speeds, effs = [], []
    for stn in stations:
        num = int(res.get(f"num_pumps_{stn.lower()}",0))
        if num > 0:
            speeds.append(res.get(f"speed_{stn.lower()}",0))
            effs.append(res.get(f"efficiency_{stn.lower()}",0))
    avg_speed = np.mean(speeds) if speeds else 0
    avg_eff   = np.mean(effs)   if effs   else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Cost (INR)", f"₹{res.get('total_cost',0):,.2f}")
    c2.metric("Total Pumps", sum(int(res.get(f"num_pumps_{s.lower()}",0)) for s in stations))
    c3.metric("Avg Speed (rpm)", f"{avg_speed:.2f}")
    c4.metric("Avg Pumping Efficiency (%)", f"{avg_eff:.2f}")

    # Summary DataFrame
    summary = {"Process Particulars":[
        "Power & Fuel cost (INR/day)","DRA cost (INR/day)","No. of Pumps",
        "Pump Speed (rpm)","Pump Efficiency (%)","Reynold's No.",
        "Dynamic Head Loss (m)","Velocity (m/s)","Residual Head (m)",
        "SDH (m)","Drag Reduction (%)"
    ]}
    for stn in stations:
        key = stn.lower()
        num = int(res.get(f"num_pumps_{key}",0))
        sp = res.get(f"speed_{key}",0) if num > 0 else 0
        ef = res.get(f"efficiency_{key}",0) if num > 0 else 0
        summary[stn] = [
            round(res.get(f"power_cost_{key}",0),2),
            round(res.get(f"dra_cost_{key}",0),2),
            num,
            round(sp,2),
            round(ef,2),
            round(res.get(f"reynolds_{key}",0),2),
            round(res.get(f"head_loss_{key}",0),2),
            round(res.get(f"velocity_{key}",0),2),
            round(res.get(f"residual_head_{key}",0),2),
            round(res.get(f"sdh_{key}",0),2),
            round(res.get(f"drag_reduction_{key}",0),2)
        ]
    df_sum = pd.DataFrame(summary)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summary Table", "💰 Cost Charts", "⚙️ Performance Charts",
        "🌀 System Curves", "🔄 Pump-System Interaction"
    ])

    # Tab 1: Summary + Downloads
    with tab1:
        st.markdown("<div class='section-title'>Optimized Parameters Summary</div>", unsafe_allow_html=True)
        fmt = {col:"{:.0f}" if col=="No. of Pumps" else "{:.2f}" for col in df_sum.columns if col!="Process Particulars"}
        st.dataframe(df_sum.style.format(fmt).set_properties(**{'text-align':'center'}), use_container_width=True)
        csv_bytes = df_sum.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Summary as CSV", csv_bytes, "pipeline_summary.csv", "text/csv")

        # PDF Report
        try:
            from fpdf import FPDF
            pdf = FPDF('L', 'mm', 'A4')
            pdf.set_auto_page_break(True, margin=15)
            page_w, page_h = pdf.w, pdf.h

            # Cover Page
            pdf.add_page()
            pdf.set_font('Arial', 'B', 20)
            pdf.cell(0, 12, 'Optimized Report', ln=1, align='C')
            pdf.ln(4)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, 'Input Parameters:', ln=1)
            for k, v in {
                'Flow (m³/hr)': FLOW,
                'Viscosity (cSt)': KV,
                'Density (kg/m³)': rho,
                'SFC Jamnagar (gm/bhp/hr)': SFC_J,
                'SFC Rajkot (gm/bhp/hr)': SFC_R,
                'SFC Surendranagar (gm/bhp/hr)': SFC_S,
                'DRA Rate (INR/L)': RateDRA,
                'HSD Rate (INR/L)': Price_HSD
            }.items():
                pdf.cell(0, 6, f"{k}: {v}", ln=1)
            pdf.ln(6)

            # Summary Table Page
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Summary Table', ln=1, align='L')
            pdf.ln(2)
            pdf.set_font('Arial', '', 10)
            cols = df_sum.columns.tolist()
            col_w = page_w / len(cols) - 4
            for col in cols:
                pdf.cell(col_w, 6, col, border=1, align='C')
            pdf.ln()
            for _, row in df_sum.iterrows():
                for col in cols:
                    pdf.cell(col_w, 6, str(row[col]), border=1, align='C')
                pdf.ln()

            def add_chart(fig):
                img = fig.to_image(format='png', width=int(page_w*0.8), height=int(page_h*0.4))
                x = (page_w - page_w*0.8) / 2
                y = pdf.get_y() + 4
                pdf.image(BytesIO(img), x=x, y=y, w=page_w*0.8)
                pdf.ln(page_h*0.4 + 4)
                if pdf.get_y() > page_h - 50:
                    pdf.add_page()

            # Charts for PDF
            dfc = pd.DataFrame({
                'Station': stations,
                'Power & Fuel (INR/day)': [res.get(f"power_cost_{s.lower()}",0) for s in stations],
                'DRA (INR/day)': [res.get(f"dra_cost_{s.lower()}",0) for s in stations]
            })
            add_chart(px.bar(
                dfc.melt(id_vars='Station', value_vars=['Power & Fuel (INR/day)','DRA (INR/day)'],
                var_name='Type', value_name='Amount'),
                x='Station', y='Amount', color='Type', barmode='group',
                title='Cost Breakdown per Station'
            ))
            add_chart(px.bar(
                pd.DataFrame({
                    'Station': stations,
                    'Head Loss (m)': [res.get(f"head_loss_{s.lower()}",0) for s in stations]
                }),
                x='Station', y='Head Loss (m)', title='Head Loss by Station'
            ))

            flow_arr = np.arange(0,4501,100)
            for stn in stations[:4]:
                key = stn.lower()
                A, B, C = res.get(f"coef_A_{key}"), res.get(f"coef_B_{key}"), res.get(f"coef_C_{key}")
                dol, mn = res.get(f"dol_{key}"), res.get(f"min_rpm_{key}")
                # Pump curves
                curves = []
                for rpm in np.arange(mn, dol+1, 100):
                    H = (A*flow_arr**2 + B*flow_arr + C) * (rpm/dol)**2
                    curves.append(pd.DataFrame({'Flow (m³/hr)': flow_arr, 'Head (m)': H, 'RPM': rpm}))
                add_chart(px.line(pd.concat(curves), x='Flow (m³/hr)', y='Head (m)', color='RPM', title=f'Pump Curves ({stn})'))
                # Efficiency curves
                eff_curves = []
                for rpm in np.arange(mn, dol+1, 100):
                    eq = flow_arr * dol / rpm
                    P, Q, R, S, T = [res.get(f"coef_{x}_{key}") for x in ['P','Q','R','S','T']]
                    E = (P*eq**4 + Q*eq**3 + R*eq**2 + S*eq + T) / 100
                    if (mask := E>0).any():
                        eff_curves.append(pd.DataFrame({'Flow (m³/hr)': flow_arr[mask], 'Efficiency (%)': E[mask], 'RPM': rpm}))
                if eff_curves:
                    add_chart(px.line(pd.concat(eff_curves), x='Flow (m³/hr)', y='Efficiency (%)', color='RPM', title=f'Efficiency Curves ({stn})'))

            # System & Interaction
            for stn in stations[:-1]:
                p, sd = params[stn], static_heads[stn]
                # System curves
                sys_curves = []
                for dra in range(0,45,5):
                    v = flow_arr/(3.414*p['d']**2/4)/3600
                    Re = v*p['d']/(KV*1e-6)
                    f = 0.25/(np.log10((p['e']/p['d']/3.7)+(5.74/(Re**0.9)))**2)
                    sys_curves.append(pd.DataFrame({
                        'Flow (m³/hr)': flow_arr,
                        'SDHR (m)': sd + f*(p['L']*1000/p['d'])*(v**2/(2*9.81))*(1 - dra/100),
                        'DRA (%)': dra
                    }))
                add_chart(px.line(pd.concat(sys_curves), x='Flow (m³/hr)', y='SDHR (m)', color='DRA (%)', title=f'System Curves ({stn})'))

                # Interaction curves
                psys = []
                num = int(res.get(f"num_pumps_{stn.lower()}",1))
                A, B, C = res.get(f"coef_A_{stn.lower()}"), res.get(f"coef_B_{stn.lower()}"), res.get(f"coef_C_{stn.lower()}")
                for dra in range(0,41,5):
                    v = flow_arr/(3.414*p['d']**2/4)/3600
                    Re = v*p['d']/(KV*1e-6)
                    f = 0.25/(np.log10((p['e']/p['d']/3.7)+(5.74/(Re**0.9)))**2)
                    psys.append(pd.DataFrame({
                        'Flow (m³/hr)': flow_arr,
                        'Head (m)': sd + f*(p['L']*1000/p['d'])*(v**2/(2*9.81))*(1 - dra/100),
                        'Curve': f'Sys DRA {dra}%'
                    }))
                for rpm in np.arange(res.get(f"min_rpm_{stn.lower()}",0), res.get(f"dol_{stn.lower()}",0)+1,100):
                    H = (A*flow_arr**2 + B*flow_arr + C) * (rpm/res.get(f"dol_{stn.lower()}",1))**2
                    psys.append(pd.DataFrame({'Flow (m³/hr)': flow_arr, 'Head (m)': H, 'Curve': f'Pump {rpm} rpm'}))
                    if num > 1:
                        psys.append(pd.DataFrame({'Flow (m³/hr)': flow_arr, 'Head (m)': H*num, 'Curve': f'Pump Total {rpm} x{num}'}))
                add_chart(px.line(pd.concat(psys), x='Flow (m³/hr)', y='Head (m)', color='Curve', title=f'Pump-System Interaction ({stn})'))

            buffer = BytesIO()
            pdf.output(buffer)
            st.download_button("📥 Download Full PDF Report", buffer.getvalue(), "Optimized_Report.pdf", "application/pdf")

        except ModuleNotFoundError:
            st.warning("Install fpdf2 and kaleido: pip install fpdf2 kaleido")

    # Tab 2: Cost Charts
    with tab2:
        st.markdown("<div class='section-title'>Cost Breakdown per Station</div>", unsafe_allow_html=True)
        df_cost = pd.DataFrame({
            "Station": stations,
            "Power & Fuel (INR/day)": [res.get(f"power_cost_{s.lower()}",0) for s in stations],
            "DRA (INR/day)": [res.get(f"dra_cost_{s.lower()}",0) for s in stations]
        })
        fig2 = px.bar(
            df_cost.melt(id_vars="Station", value_vars=["Power & Fuel (INR/day)", "DRA (INR/day)"],
                         var_name="Type", value_name="Amount"),
            x="Station", y="Amount", color="Type", barmode="group", title="Cost Components by Station"
        )
        fig2.update_layout(xaxis_title="Station", yaxis_title="Amount (INR)")
        fig2.update_yaxes(tickformat=".2f")
        st.plotly_chart(fig2, use_container_width=True)

    # Tab 3: Performance Charts
    with tab3:
        perf_tab, pump_tab, eff_tab = st.tabs(["Performance Metrics", "Pump Characteristic Curves", "Pump Efficiency Curves"])
        with perf_tab:
            st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)
            df_perf = pd.DataFrame({
                "Station": stations,
                "Head Loss (m)": [res.get(f"head_loss_{s.lower()}",0) for s in stations]
            })
            fig_pm = go.Figure(go.Bar(x=df_perf["Station"], y=df_perf["Head Loss (m)"]))
            fig_pm.update_layout(title_text="Head Loss by Station", xaxis_title="Station", yaxis_title="Head Loss (m)")
            fig_pm.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig_pm, use_container_width=True)
        with pump_tab:
            st.markdown("<div class='section-title'>Pump Characteristic Curves</div>", unsafe_allow_html=True)
            sel = st.multiselect("Select stations", stations[:4], default=stations[:4])
            flow_arr_tab = np.arange(0,4501,100)
            for stn in sel:
                key = stn.lower()
                A, B, C = res.get(f"coef_A_{key}"), res.get(f"coef_B_{key}"), res.get(f"coef_C_{key}")
                dol, mn = res.get(f"dol_{key}"), res.get(f"min_rpm_{key}")
                if None in [A, B, C, dol, mn]:
                    continue
                dfs = []
                for rpm in np.arange(mn, dol+1, 100):
                    H = (A*flow_arr_tab**2 + B*flow_arr_tab + C) * (rpm/dol)**2
                    dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr_tab, "Head (m)": H, "RPM": rpm}))
                df_all = pd.concat(dfs, ignore_index=True)
                fig_pc = px.line(df_all, x="Flow (m³/hr)", y="Head (m)", color="RPM", title=f"Pump Curves ({stn})")
                fig_pc.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
                fig_pc.update_yaxes(tickformat=".2f")
                st.plotly_chart(fig_pc, use_container_width=True)
        with eff_tab:
            st.markdown("<div class='section-title'>Pump Efficiency Curves</div>", unsafe_allow_html=True)
            flow_arr_tab = np.arange(0,4501,100)
            for stn in stations[:4]:
                key = stn.lower()
                P, Q, R, S, T = [res.get(f"coef_{x}_{key}") for x in ['P','Q','R','S','T']]
                dol, mn = res.get(f"dol_{key}"), res.get(f"min_rpm_{key}")
                if None in [P, Q, R, S, T, dol, mn]:
                    continue
                dfs = []
                for rpm in np.arange(mn, dol+1, 100):
                    eq = flow_arr_tab * dol / rpm
                    E = (P*eq**4 + Q*eq**3 + R*eq**2 + S*eq + T) / 100
                    mask = E > 0
                    if not mask.any():
                        continue
                    dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr_tab[mask], "Efficiency (%)": E[mask], "RPM": rpm}))
                if dfs:
                    df_eff = pd.concat(dfs, ignore_index=True)
                    fig_eff = px.line(df_eff, x="Flow (m³/hr)", y="Efficiency (%)", color="RPM", title=f"Efficiency Curves ({stn})")
                    fig_eff.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="Efficiency (%)")
                    fig_eff.update_yaxes(tickformat=".2f")
                    st.plotly_chart(fig_eff, use_container_width=True)

    # Tab 4: System Curves of SDHR
    with tab4:
        st.markdown("<div class='section-title'>System Curves of SDHR</div>", unsafe_allow_html=True)
        flow_arr_tab = np.arange(0,4501,100)
        for stn in stations[:-1]:
            p, sd = params[stn], static_heads[stn]
            dfs = []
            for dra in range(0,45,5):
                v = flow_arr_tab/(3.414*p['d']**2/4)/3600
                Re = v*p['d']/(KV*1e-6)
                f = 0.25/(np.log10((p['e']/p['d']/3.7)+(5.74/(Re**0.9)))**2)
                DH = (f*(p['L']*1000/p['d'])*(v**2/(2*9.81))) * (1 - dra/100)
                dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr_tab, "SDHR (m)": sd+DH, "DRA (%)": dra}))
            df_sys = pd.concat(dfs, ignore_index=True)
            fig_sys = px.line(df_sys, x="Flow (m³/hr)", y="SDHR (m)", color="DRA (%)", title=f"SDHR Curves ({stn})")
            fig_sys.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="SDHR (m)")
            fig_sys.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig_sys, use_container_width=True)

    # Tab 5: Pump-System Interaction
    with tab5:
        st.markdown("<div class='section-title'>Pump-System Interaction</div>", unsafe_allow_html=True)
        flow_arr_tab = np.arange(0,4501,100)
        for stn in stations[:-1]:
            p, sd = params[stn], static_heads[stn]
            dfs = []
            # System curves
            for dra in range(0,41,5):
                v = flow_arr_tab/(3.414*p['d']**2/4)/3600
                Re = v*p['d']/(KV*1e-6)
                f = 0.25/(np.log10((p['e']/p['d']/3.7)+(5.74/(Re**0.9)))**2)
                DH = (f*(p['L']*1000/p['d'])*(v**2/(2*9.81))) * (1 - dra/100)
                dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr_tab, "Head (m)": sd+DH, "Curve": f"Sys DRA {dra}%"}))
            # Pump operation and series combinations
            A = res.get(f"coef_A_{stn.lower()}")
            B = res.get(f"coef_B_{stn.lower()}")
            C = res.get(f"coef_C_{stn.lower()}")
            min_rpm = res.get(f"min_rpm_{stn.lower()}",0)
            dol = res.get(f"dol_{stn.lower()}",1)
            for rpm in np.arange(min_rpm, dol+1, 100):
                H = (A*flow_arr_tab**2 + B*flow_arr_tab + C) * (rpm/dol)**2
                # Single pump curve
                dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr_tab, "Head (m)": H, "Curve": f"Pump {rpm} rpm"}))
                # Optimized series total (existing)
                num = int(res.get(f"num_pumps_{stn.lower()}",1))
                if num > 1:
                    dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr_tab, "Head (m)": H*num, "Curve": f"Pump Total {rpm} x{num}"}))
                # Additional fixed 2-pump series curves for specified stations
                if stn in ["Vadinar","Jamnagar","Rajkot","Surendranagar"]:
                    dfs.append(pd.DataFrame({"Flow (m³/hr)": flow_arr_tab, "Head (m)": H*2, "Curve": f"2 pumps in series {rpm} rpm"}))
            df_plot = pd.concat(dfs, ignore_index=True)
            fig_int = px.line(df_plot, x="Flow (m³/hr)", y="Head (m)", color="Curve", title=f"Pump-System Interaction ({stn})")
            # Increase figure size
            fig_int.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)", height=600)
            fig_int.update_yaxes(tickformat=".2f")
            st.plotly_chart(fig_int, use_container_width=True, height=600)

    st.markdown("---")
    st.caption("© 2025 Developed by Parichay Das. All rights reserved.")
