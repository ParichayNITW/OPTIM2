import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import random

st.set_page_config(page_title="Branched Pipeline Visual Builder", layout="wide")
st.title("🛠️ Branched Pipeline Visual Builder Prototype")

if 'nodes' not in st.session_state:
    st.session_state['nodes'] = [
        Node(id="1", label="Station 1", size=25, shape="circle", color="green")
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

# Show the graph and let user select a node
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

# Gather station data for each node
st.subheader("Station and Pipe Segment Data Entry")
station_params = {}
for node in st.session_state['nodes']:
    stn_id = node.id
    parent = st.session_state['parent_map'][stn_id]
    with st.expander(f"{node.label}", expanded=False):
        stn_name = st.text_input(f"Name (Station {stn_id})", value=node.label, key=f"name_{stn_id}")
        elev = st.number_input("Elevation (m)", value=0.0, step=1.0, key=f"elev_{stn_id}")
        is_pump = st.checkbox("Is this a Pump Station?", key=f"pump_{stn_id}")
        flow_demand = st.number_input("Flow Demand at this Node (m³/hr)", value=0.0, step=1.0, key=f"fd_{stn_id}")
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
            "L": L,
            "D": D,
            "parent": parent
        }

st.markdown("---")

if st.button("🧮 Compute Branch Flows & Head (Demo Only)"):
    # For now, just print the structure and demands as a tree
    st.subheader("Network Structure & Demands")
    for stn_id, params in station_params.items():
        st.write(f"Station {params['name']} | Parent: {params['parent']} | "
                 f"Pump: {params['is_pump']} | Demand: {params['flow_demand']} m³/hr | "
                 f"L: {params['L']} km | D: {params['D']} m")
    st.info("This is a prototype. The next step is to pass this tree/graph structure to a recursive hydraulic model backend for full optimization.")

st.warning("Prototype: No actual hydraulics/optimization yet. Next step is to develop graph-based hydraulic solver backend.")

st.markdown("---")
st.caption("© 2025 (R) Parichay Das. All rights reserved.")
