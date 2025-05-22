# branched_pipeline_app.py

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd

# =========================
# Backend logic
# =========================

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
        flow = node['total_flow']
        L = node.get('L', 1)
        D = node.get('D', 0.7)
        rough = node.get('rough', 0.00004)
        # Toy head loss; replace with real formula for production use!
        K = 0.01
        head_loss = K * (flow ** 1.8) / (D ** 4.8)
        head_loss = head_loss * L
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
            'Station': node['name'],
            'Parent': node['parent'],
            'Flow_Demand (m³/hr)': node.get('flow_demand', 0),
            'Total_Flow (m³/hr)': node.get('total_flow', 0),
            'Required_Head (m)': round(node.get('required_head', 0), 2),
            'Is_Pump': "Yes" if node.get('is_pump', False) else "No",
        })
    return pd.DataFrame(results)

# =========================
# Streamlit Frontend
# =========================

st.set_page_config(page_title="Branched Pipeline Optimizer", layout="wide")
st.title("🌲 Branched Pipeline Optimizer")

st.markdown("""
This tool lets you build a pipeline network (with branches), specify station/pipe properties, and run a hydraulic/flow analysis. Later, you can extend this to full cost optimization.
""")

# ----- Initialize session state -----
if 'nodes' not in st.session_state:
    st.session_state['nodes'] = [
        Node(id="1", label="Station 1", size=30, shape="circle", color="green")
    ]
    st.session_state['edges'] = []
    st.session_state['next_id'] = 2
    st.session_state['parent_map'] = {"1": None}  # Node ID to parent mapping

def add_station(parent_id, branch=False):
    n_id = str(st.session_state['next_id'])
    label = f"Station {n_id}" + (" (Branch)" if branch else "")
    node = Node(id=n_id, label=label, size=25, shape="circle", color="orange" if branch else "blue")
    st.session_state['nodes'].append(node)
    st.session_state['edges'].append(Edge(source=parent_id, target=n_id, label="Branch" if branch else "Mainline"))
    st.session_state['next_id'] += 1
    st.session_state['parent_map'][n_id] = parent_id

# --- Graphical Builder ---
st.subheader("1️⃣ Build Your Pipeline Network")
selected = agraph(
    nodes=st.session_state['nodes'],
    edges=st.session_state['edges'],
    config=Config(
        directed=True, 
        nodeHighlightBehavior=True, 
        highlightColor="#F7A7A6", 
        collapsible=True,
        width=900,
        height=400,
    ),
)

if selected:
    st.success(f"Selected node: {selected}")
    parent_id = selected
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Add Mainline Station to {selected}"):
            add_station(parent_id, branch=False)
    with col2:
        if st.button(f"Add Branch Station to {selected}"):
            add_station(parent_id, branch=True)

st.markdown("---")

# --- Node Data Editing ---
st.subheader("2️⃣ Station and Pipe Segment Data Entry")
station_params = {}
for node in st.session_state['nodes']:
    stn_id = node.id
    parent = st.session_state['parent_map'][stn_id]
    with st.expander(f"{node.label}", expanded=False):
        stn_name = st.text_input(f"Name (Station {stn_id})", value=node.label, key=f"name_{stn_id}")
        elev = st.number_input("Elevation (m)", value=0.0, step=1.0, key=f"elev_{stn_id}")
        is_pump = st.checkbox("Is this a Pump Station?", key=f"pump_{stn_id}")
        flow_demand = st.number_input("Flow Demand at this Node (m³/hr)", value=0.0, step=1.0, key=f"fd_{stn_id}")
        min_head = st.number_input("Minimum Required Head at Node (m)", value=10.0, step=0.1, key=f"mh_{stn_id}")
        if parent:
            L = st.number_input("Pipe Length from parent (km)", value=20.0, step=0.1, key=f"L_{stn_id}")
            D = st.number_input("Pipe Diameter (m)", value=0.7, step=0.01, key=f"D_{stn_id}")
        else:
            L = None
            D = None
        station_params[stn_id] = {
            "name": stn_name,
            "elev": elev,
            "is_pump": is_pump,
            "flow_demand": flow_demand,
            "min_head": min_head,
            "L": L,
            "D": D,
            "parent": parent,
            "children": []
        }

# --- Build children lists (from edges) ---
for edge in st.session_state['edges']:
    src = edge.source
    tgt = edge.target
    station_params[src].setdefault("children", []).append(tgt)

# --- Backend calculation ---
st.markdown("---")
st.subheader("3️⃣ Hydraulic Analysis")
if st.button("🧮 Compute Flows & Heads"):
    stations = station_params
    # Calculate flows (from every leaf)
    for node_id, node in stations.items():
        if not node.get('children'):
            calculate_flows(stations, node_id)
    # Find root node (parent == None), propagate heads
    root_id = next(i for i, stn in stations.items() if stn['parent'] is None)
    calculate_heads(stations, root_id)
    # Show results
    df_results = get_branch_results(stations)
    st.markdown("### 🚩 Results Table")
    st.dataframe(df_results, use_container_width=True)
    st.success("Flows and required heads computed for every station. You can now optimize pump assignments, run more advanced simulation, or export this network.")
    st.download_button("Download Results as CSV", df_results.to_csv(index=False), file_name="branched_pipeline_results.csv")
else:
    st.info("Build your network and enter parameters, then click 'Compute Flows & Heads'.")

st.markdown("---")
st.caption("© 2025 (R) Parichay Das. All rights reserved.")
