# branched_pipeline_model.py

def calculate_flows(stations, node_id):
    """
    Recursively calculate flow at each node.
    Fills node['total_flow'] for every node.
    """
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
    """
    Recursively calculate required head at each node, moving upstream.
    downstream_head: minimum required head at the node just downstream (m)
    Fills node['required_head'] for every node.
    """
    node = stations[node_id]
    if node['parent'] is not None:
        # Compute head loss in parent->this segment (simplified Hazen-Williams or Darcy–Weisbach for demo)
        flow = node['total_flow']
        L = node.get('L', 1)
        D = node.get('D', 0.7)
        rough = node.get('rough', 0.00004)
        # --- Toy formula for head loss (real models would use Darcy-Weisbach etc.) ---
        # Example: hL = K * Q^1.8 / D^4.8
        K = 0.01
        head_loss = K * (flow ** 1.8) / (D ** 4.8)
        head_loss = head_loss * L
    else:
        head_loss = 0

    # Minimum required head at this node (can be user input, here fixed to 10m)
    min_node_head = node.get('min_head', 10)
    node['required_head'] = downstream_head + head_loss + min_node_head

    # Propagate upstream
    if node['parent']:
        calculate_heads(stations, node['parent'], node['required_head'])

def get_branch_results(stations):
    """
    Returns a list of results (dicts) for all nodes for display.
    """
    results = []
    for node_id, node in stations.items():
        results.append({
            'Station': node['name'],
            'Parent': node['parent'],
            'Flow_Demand': node.get('flow_demand', 0),
            'Total_Flow': node.get('total_flow', 0),
            'Required_Head': node.get('required_head', 0),
            'Is_Pump': node.get('is_pump', False),
        })
    return results

# Example usage (this would come from your frontend!)
if __name__ == "__main__":
    # Simple test: Station 1 -> Station 2 -> Station 3 (mainline), Station 2 -> 2A (branch)
    stations = {
        "1": {"name": "Station 1", "elev": 0, "is_pump": True, "flow_demand": 0, "L": None, "D": None, "parent": None, "children": ["2"]},
        "2": {"name": "Station 2", "elev": 10, "is_pump": True, "flow_demand": 0, "L": 50, "D": 0.7, "parent": "1", "children": ["3", "2A"]},
        "3": {"name": "Station 3", "elev": 15, "is_pump": False, "flow_demand": 50, "L": 40, "D": 0.7, "parent": "2", "children": []},
        "2A": {"name": "Station 2A", "elev": 8, "is_pump": False, "flow_demand": 20, "L": 30, "D": 0.5, "parent": "2", "children": []}
    }
    # 1. Calculate flows (start from every leaf)
    for node_id, node in stations.items():
        if not node.get('children'):
            calculate_flows(stations, node_id)
    # 2. Find root, propagate heads
    root_id = next(i for i, stn in stations.items() if stn['parent'] is None)
    calculate_heads(stations, root_id)
    # 3. Print results
    for res in get_branch_results(stations):
        print(res)
