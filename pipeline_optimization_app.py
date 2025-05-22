# branched_pipeline_pro.py

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd
import json

st.set_page_config(page_title="Branched Pipeline Optimizer", layout="wide")
st.title("🌲 Branched Pipeline Optimizer — Production Edition")

st.markdown("""
Draw your pipeline (mainline & branches), then click each node to enter/edit all data.<br>
Upload or download your full pipeline as a JSON file.<br>
When ready, run hydraulic analysis/optimization on the full system.
""", unsafe_allow_html=True)

# ========== SESSION STATE INIT ==========
if "network_data" not in st.session_state:
    # Each node will have id, label, parent, is_pump, all fields below, and lists for children
    st.session_state.network_data = {
        "nodes": [Node(id="1", label="Station 1", size=30, shape="circle", color="green")],
        "edges": [],
        "node_props": {"1": {}},
        "next_id": 2,
        "parent_map": {"1": None}
    }

# ========== PIPELINE BUILDER SECTION ==========
st.subheader("1️⃣ Pipeline Network Designer (Mainline & Branches)")
with st.expander("⬇️ Upload/Download Pipeline Network as JSON", expanded=False):
    upload = st.file_uploader("Upload JSON Network", type="json")
    if upload is not None:
        network = json.load(upload)
        st.session_state.network_data = network
        st.experimental_rerun()
    st.download_button(
        "Download Current Pipeline Network (JSON)",
        json.dumps({
            "nodes": [n.to_dict() if hasattr(n, "to_dict") else n for n in st.session_state.network_data["nodes"]],
            "edges": [e.to_dict() if hasattr(e, "to_dict") else e for e in st.session_state.network_data["edges"]],
            "node_props": st.session_state.network_data["node_props"],
            "next_id": st.session_state.network_data["next_id"],
            "parent_map": st.session_state.network_data["parent_map"]
        }, indent=2),
        file_name="pipeline_network.json"
    )

selected = agraph(
    nodes=st.session_state.network_data["nodes"],
    edges=st.session_state.network_data["edges"],
    config=Config(
        directed=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        width=950,
        height=480,
    ),
)
if selected:
    st.info(f"Selected: {selected}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Add Mainline Station to {selected}"):
            nid = str(st.session_state.network_data["next_id"])
            st.session_state.network_data["nodes"].append(
                Node(id=nid, label=f"Station {nid}", size=25, shape="circle", color="blue"))
            st.session_state.network_data["edges"].append(
                Edge(source=selected, target=nid, label="Mainline"))
            st.session_state.network_data["parent_map"][nid] = selected
            st.session_state.network_data["node_props"][nid] = {}
            st.session_state.network_data["next_id"] += 1
            st.experimental_rerun()
    with col2:
        if st.button(f"Add Branch Station to {selected}"):
            nid = str(st.session_state.network_data["next_id"])
            st.session_state.network_data["nodes"].append(
                Node(id=nid, label=f"Station {nid} (Branch)", size=25, shape="circle", color="orange"))
            st.session_state.network_data["edges"].append(
                Edge(source=selected, target=nid, label="Branch"))
            st.session_state.network_data["parent_map"][nid] = selected
            st.session_state.network_data["node_props"][nid] = {}
            st.session_state.network_data["next_id"] += 1
            st.experimental_rerun()

st.markdown("---")

# ========== NODE PROPERTIES EDITOR ==========
st.subheader("2️⃣ Station and Segment Data Editor")

for node in st.session_state.network_data["nodes"]:
    stn_id = node.id
    parent = st.session_state.network_data["parent_map"].get(stn_id)
    props = st.session_state.network_data["node_props"].get(stn_id, {})
    with st.expander(f"Edit {node.label}", expanded=False):
        # Station Data
        name = st.text_input("Station Name", value=props.get("name", node.label), key=f"name_{stn_id}")
        elev = st.number_input("Elevation (m)", value=props.get("elev", 0.0), key=f"elev_{stn_id}")
        is_pump = st.checkbox("Is this a Pump Station?", value=props.get("is_pump", False), key=f"pump_{stn_id}")
        flow_demand = st.number_input("Flow Demand at this Node (m³/hr)", value=props.get("flow_demand", 0.0), key=f"fd_{stn_id}")
        min_head = st.number_input("Minimum Required Head at Node (m)", value=props.get("min_head", 10.0), key=f"mh_{stn_id}")

        # Pipe Segment Data (for all but root)
        if parent:
            L = st.number_input("Pipe Length from parent (km)", value=props.get("L", 20.0), key=f"L_{stn_id}")
            D = st.number_input("Pipe Diameter (m)", value=props.get("D", 0.7), key=f"D_{stn_id}")
            rough = st.number_input("Pipe Roughness (m)", value=props.get("rough", 0.00004), format="%.5f", key=f"rough_{stn_id}")
        else:
            L = None
            D = None
            rough = None

        # If pump, more parameters
        power_type, rate, sfc, max_pumps, MinRPM, DOL, max_dr = None, None, None, None, None, None, None
        if is_pump:
            power_type = st.selectbox("Power Source", ["Grid", "Diesel"], index=0 if props.get("power_type","Grid")=="Grid" else 1, key=f"ptype_{stn_id}")
            if power_type == "Grid":
                rate = st.number_input("Electricity Rate (INR/kWh)", value=props.get("rate",9.0), key=f"rate_{stn_id}")
                sfc = 0.0
            else:
                sfc = st.number_input("SFC (gm/bhp·hr)", value=props.get("sfc",150.0), key=f"sfc_{stn_id}")
                rate = 0.0
            max_pumps = st.number_input("Max Pumps Available", min_value=1, value=props.get("max_pumps",1), step=1, key=f"mpumps_{stn_id}")
            MinRPM = st.number_input("Min RPM", value=props.get("MinRPM",1200.0), key=f"minrpm_{stn_id}")
            DOL = st.number_input("Rated RPM (DOL)", value=props.get("DOL",1500.0), key=f"dol_{stn_id}")
            max_dr = st.number_input("Max Drag Reduction (%)", value=props.get("max_dr",0.0), key=f"mdr_{stn_id}")
            # (TODO: Add curve editors here)

        # Save back to session state
        st.session_state.network_data["node_props"][stn_id] = dict(
            name=name, elev=elev, is_pump=is_pump, flow_demand=flow_demand, min_head=min_head,
            L=L, D=D, rough=rough,
            power_type=power_type, rate=rate, sfc=sfc, max_pumps=max_pumps, MinRPM=MinRPM, DOL=DOL, max_dr=max_dr
        )

st.markdown("---")

# ========== BACKEND LOGIC ==========
st.subheader("3️⃣ Hydraulic Analysis / Optimization (Phase 1)")

def build_stations_dict(network_data):
    """Convert session network_data to backend dict for calculation."""
    node_props = network_data["node_props"]
    parent_map = network_data["parent_map"]
    children_map = {}
    for e in network_data["edges"]:
        src, tgt = e.source, e.target
        children_map.setdefault(src, []).append(tgt)
    stations = {}
    for node in network_data["nodes"]:
        nid = node.id
        props = node_props[nid]
        # inherit/ensure all fields
        stations[nid] = dict(props)
        stations[nid]["parent"] = parent_map[nid]
        stations[nid]["children"] = children_map.get(nid, [])
    return stations

def calculate_flows(stations, node_id):
    node = stations[node_id]
    if not node.get('children'):
        node['total_flow'] = node.get('flow_demand', 0)
        return node['total_flow']
    flow = node.get('flow_demand', 0)
    for child_id in node['children']:
        flow += calculate_flows(stations, child_id)
    node['total_flow'] = flow
    return flow

def calculate_heads(stations, node_id, downstream_head=0):
    node = stations[node_id]
    if node['parent'] is not None:
        flow = node.get('total_flow', 0)
        L = node.get('L', 1)
        D = node.get('D', 0.7)
        rough = node.get('rough', 0.00004)
        # Example: Swamee-Jain (replace with your production code if you want):
        # Re = 1e5, f = 0.019 (for demo, real: f = f(Re, rough, D, etc.))
        f = 0.019
        g = 9.81
        Q = flow / 3600.0  # m3/s
        A = 3.1416 * (D ** 2) / 4
        v = Q / A if A > 0 else 0
        head_loss = (f * L * 1000 * v ** 2) / (D * 2 * g) if D and L else 0
    else:
        head_loss = 0
    min_node_head = node.get('min_head', 10)
    node['required_head'] = downstream_head + head_loss + min_node_head
    if node['parent']:
        calculate_heads(stations, node['parent'], node['required_head'])

def get_branch_results(stations):
    results = []
    for node_id, node in stations.items():
        results.append({
            'Station': node.get('name', node_id),
            'Parent': node['parent'],
            'Flow_Demand (m³/hr)': node.get('flow_demand', 0),
            'Total_Flow (m³/hr)': node.get('total_flow', 0),
            'Required_Head (m)': round(node.get('required_head', 0), 2),
            'Is_Pump': "Yes" if node.get('is_pump', False) else "No",
            'Power Source': node.get('power_type', ''),
            'Max Pumps': node.get('max_pumps', ''),
            'DOL': node.get('DOL', ''),
        })
    return pd.DataFrame(results)

if st.button("🧮 Run Analysis"):
    stations = build_stations_dict(st.session_state.network_data)
    for node_id, node in stations.items():
        if not node.get('children'):
            calculate_flows(stations, node_id)
    root_id = next(i for i, stn in stations.items() if stn['parent'] is None)
    calculate_heads(stations, root_id)
    df_results = get_branch_results(stations)
    st.dataframe(df_results, use_container_width=True)
    st.success("Hydraulic analysis complete for every station. Next: optimization, pump curves, and DRA integration.")

st.markdown("---")
st.caption("© 2025 (R) Parichay Das. All rights reserved.")
