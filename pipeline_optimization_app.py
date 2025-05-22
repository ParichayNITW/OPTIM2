import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

if "nodes" not in st.session_state:
    st.session_state["nodes"] = [Node(id="1", label="Station 1", size=30, color="green")]
    st.session_state["edges"] = []
    st.session_state["next_id"] = 2

st.title("Branching Test")

selected = agraph(
    nodes=st.session_state["nodes"],
    edges=st.session_state["edges"],
    config=Config(directed=True, width=900, height=400),
)

if selected:
    st.success(f"Selected node: {selected}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Add Mainline Station to {selected}"):
            nid = str(st.session_state["next_id"])
            st.session_state["nodes"].append(Node(id=nid, label=f"Station {nid}", size=25, color="blue"))
            st.session_state["edges"].append(Edge(source=selected, target=nid, label="Mainline"))
            st.session_state["next_id"] += 1
            st.experimental_rerun()
    with col2:
        if st.button(f"Add Branch Station to {selected}"):
            nid = str(st.session_state["next_id"])
            st.session_state["nodes"].append(Node(id=nid, label=f"Station {nid} (Branch)", size=25, color="orange"))
            st.session_state["edges"].append(Edge(source=selected, target=nid, label="Branch"))
            st.session_state["next_id"] += 1
            st.experimental_rerun()
