import json
from param_bench.train.compute.python.tools.execution_graph import ExecutionGraph

def read_dictionary_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


def find_node(nodes, target_name):
    target_node = None
    found = False  # Flag variable to track search status

    def traverse(root, target_name):
        nonlocal target_node, found

        if found:  # If target node is already found, exit immediately
            return

        for node in root.children:
            if node.name == target_name:
                target_node = node
                found = True  # Set the flag to indicate target node is found
                return  # Immediately return when target node is found

            traverse(node, target_name)

    root = nodes[1]
    traverse(root, target_name)
    return target_node


def find_node_with_children(nodes, target_node_name, child_name):
    target_node = None
    found = False  # Flag variable to track search status

    def traverse(root, target_node_name, child_name):
        nonlocal target_node, found

        if found:  # If target node is already found, exit immediately
            return

        if root.name == target_node_name:
            for child in root.children:
                if child_name in child.name.lower():
                    target_node = root
                    found = True
                    return

        for child in root.children:
            traverse(child, target_node_name, child_name)

    root = nodes[1]
    traverse(root, target_node_name, child_name)
    return target_node


def collect_nodes(node):
    def traverse(node):
        nonlocal nodes
        nodes.append(node)
        for child in node.children:
            traverse(child)

    nodes = []
    traverse(node)
    nodes = nodes[1:]
    sorted_nodes = sorted(nodes, key=lambda x: x.id)
    return sorted_nodes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Link kineto trace with execution trace")
    parser.add_argument("--et_file", type=str, required=True, help="Path to the execution trace file")
    parser.add_argument("--kineto_file", type=str, required=True, help="Path to the kineto trace file")
    parser.add_argument("--annotation", type=str, required=True, help="User annotation of the iteration step of execution trace")

    args = parser.parse_args()

    # Analysis of kineto trace
    kineto_trace_events = read_dictionary_from_json(args.kineto_file)['traceEvents']

    annotation_event = [event for event in kineto_trace_events if 'name' in event and event['name'] == args.annotation]
    assert(len(annotation_event) == 1)
    sorted_kineto_trace_events = sorted(kineto_trace_events, key=lambda kv: kv['ts'])

    kineto_trace_events_under_annotation = [event for event in sorted_kineto_trace_events if event['ts'] > annotation_event[0]['ts'] and event['ts'] < annotation_event[0]['ts'] + annotation_event[0]['dur']]
    kineto_et_events = [event for event in kineto_trace_events_under_annotation if event['cat'] == 'cpu_op' or event['cat'] == 'user_annotation']

    print('Number of captured ops in kineto trace: ', len(kineto_et_events))

    # Analysis of execution trace
    with open(args.et_file, 'r') as f:
        et = ExecutionGraph(json.load(f))

    nodes = et.get_nodes(clean=True)
    annotation_node = find_node(nodes, args.annotation)

    # Backward ops in ET are not under the user annotation node (e.g., iteration#xxx), so we need to find their root node first
    backward_thread_node = find_node_with_children(nodes, '[pytorch|profiler|execution_graph|thread]', 'backward')

    # Redirect the parent of the backward nodes to the user annotation node
    for child in backward_thread_node.children:
        child.parent = annotation_node
        annotation_node.children.append(child)

    et_nodes = collect_nodes(annotation_node)

    print('Number of captured ops in execution trace: ', len(et_nodes))
    
    # Duration of et nodes
    et_node_dur = {}

    # Link kineto trace and execution trace
    for i in range(len(et_nodes)):
        et_node = et_nodes[i]
        kineto_et_event = kineto_et_events[i]
        if et_node.name != kineto_et_event['name']:
            print('Ops mismatch between kineto and execution trace:')
            print(f'Op index: {i}, kineto op name: {kineto_et_event["name"]}, kineto op timestamp: {kineto_et_event["ts"]}, \
                    execution trace op name: {et_node.name}, execution trace op id: {et_node.id}')
            exit(0)
        else:
            et_node_dur[et_node.id] = kineto_et_event['dur']

    # Add duration time to each et node and dump as et_plus
    with open(args.et_file, 'r') as f:
        et = json.load(f)
        for node in et['nodes']:
            if node['id'] in et_node_dur:
                node['dur'] = et_node_dur[node['id']]
    
    et_plus_file = args.et_file.replace('.json', '_plus.json')
    with open(et_plus_file, 'w') as f:
        json.dump(et, f, indent=4)