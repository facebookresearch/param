import networkx as nx
from networkx.algorithms import isomorphism
import json
from param_bench.train.compute.python.tools.execution_graph import ExecutionGraph
import sys


# Increase recursion limit
sys.setrecursionlimit(10**6)


def read_dictionary_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


# Add and sort ET nodes from the execution graph  
def collect_nodes(node):
    def traverse(node):
        nonlocal nodes
        nodes.append(node)
        for child in node.children:
            traverse(child)

    nodes = []
    traverse(node)
    sorted_nodes = sorted(nodes, key=lambda x: x.id)
    return sorted_nodes


class Kineto_node:
    def __init__(self, name, start, end, id):
        self.name = name
        self.start = start
        self.end = end
        self.id = id
        self.children = []


# Function to transform your self-defined tree to a directed graph with max depth
def transform_to_graph_depth(node, max_depth=100):
    graph = nx.DiGraph()
    add_node_to_graph_depth(node, graph, max_depth, 1)
    return graph


# Helper function to recursively add nodes and edges to the graph with max depth
def add_node_to_graph_depth(node, graph, max_depth, cur_depth):
    graph.add_node(node.id, label=node.name)
    if cur_depth == max_depth:
        return
    for child in node.children:
        graph.add_node(node.id, label=node.name)
        add_node_to_graph_depth(child, graph, max_depth, cur_depth + 1)


# Custom node comparison function for edit distance
def node_compare(n1, n2):
    return n1 == n2


# Find the segment that has a length closest to the target
def find_closest_segment(segs, target_length):
    closest_length = float('inf')
    closest_seg = None

    for seg in segs:
        length_difference = abs(len(seg) - target_length)
        if length_difference < closest_length:
            closest_length = length_difference
            closest_seg = seg

    return closest_seg


def exact_match(kineto_et_events, et_nodes):
    # Since kineto trace is missing the annotations for processes/threads, we add them back to match with ET
    kineto_event_per_thread = {}

    process_end_time = -1
    for i in range(len(kineto_et_events)):
        event = kineto_et_events[i]
        if event['tid'] not in kineto_event_per_thread:
            kineto_event_per_thread[event['tid']] = {}
            kineto_event_per_thread[event['tid']]['ts'] = event['ts']
            kineto_event_per_thread[event['tid']]['dur'] = event['ts'] + event['dur']
            kineto_event_per_thread[event['tid']]['index'] = i
        else:
            kineto_event_per_thread[event['tid']]['dur'] = max(kineto_event_per_thread[event['tid']]['dur'], event['ts'] + event['dur'])
        process_end_time = max(process_end_time, event['ts'] + event['dur'])

    process_event = {'name': '[pytorch|profiler|execution_graph|process]', 'ts': kineto_et_events[0]['ts'], 
                     'dur': process_end_time - kineto_et_events[0]['ts']}

    kineto_et_events.insert(0, process_event)
    
    sorted_threads = dict(sorted(kineto_event_per_thread.items(), key=lambda x: x[1]['index']))

    for index, (tid, thread_info) in enumerate(sorted_threads.items()):
        thread_event = {'name': '[pytorch|profiler|execution_graph|thread]', 'ts': thread_info['ts'], 'dur': thread_info['dur']}
        # Be careful of the insertion position, note that we already inserted process event
        kineto_et_events.insert(index + 1 + thread_info['index'], thread_event)

    # Duration of ET nodes
    et_enhanced = {}

    # Link kineto trace and execution trace
    if len(kineto_et_events) == len(et_nodes):
        for i in range(len(et_nodes)):
            et_node = et_nodes[i]
            kineto_et_event = kineto_et_events[i]
            if et_node.name == kineto_et_event['name'] or ('iteration#' in et_node.name and 'iteration#' in kineto_et_event['name']):
                et_enhanced[et_node.id] = kineto_et_event['dur']
            else:
                print('Op mismatch between kineto and execution trace:')
                print(f'Op index: {i}, kineto op name: {kineto_et_event["name"]}, kineto op timestamp: {kineto_et_event["ts"]}, ' 
                    f'execution trace op name: {et_node.name}, execution trace op id: {et_node.id}')
                exit(0)
    else:
        print('Ops count mismatch between kineto and execution trace')

    return et_enhanced


def approximate_match(kineto_et_events, et_nodes):
    # Since kineto trace is missing the annotations for processes/threads, we add them back to match with ET
    kineto_event_per_thread = {}

    # Mapping node id to the corresponding node
    kineto_nodes_mapping = {}

    for event in kineto_et_events:
        if event['tid'] not in kineto_event_per_thread:
            kineto_event_per_thread[event['tid']] = []
        kineto_event_per_thread[event['tid']].append(event)
    
    start_time = kineto_et_events[0]['ts']
    end_time = -1
    for event in kineto_et_events:
        end_time = max(end_time, event['ts'] + event['dur'])
    process_node = Kineto_node('[pytorch|profiler|execution_graph|process]', start_time, end_time, 0)
    kineto_nodes_mapping[0] = process_node

    cnt = 1
    for thread in kineto_event_per_thread:
        start_time = kineto_event_per_thread[thread][0]['ts']
        end_time = -1
        for event in kineto_event_per_thread[thread]:
            end_time = max(end_time, event['ts'] + event['dur'])

        thread_node = Kineto_node('[pytorch|profiler|execution_graph|thread]', start_time, end_time, cnt)
        kineto_nodes_mapping[cnt] = thread_node
        cnt += 1

        process_node.children.append(thread_node)

        kineto_nodes = [thread_node]
        for event in kineto_event_per_thread[thread]:
            if event['ts'] < kineto_nodes[-1].end:
                tmp = Kineto_node(event['name'], event['ts'], event['ts'] + event['dur'], cnt)
                kineto_nodes_mapping[cnt] = tmp
                cnt += 1
                kineto_nodes[-1].children.append(tmp)
                kineto_nodes.append(tmp)
            else:
                while kineto_nodes[-1].end <= event['ts']:
                    kineto_nodes.pop()
                tmp = Kineto_node(event['name'], event['ts'], event['ts'] + event['dur'], cnt)
                kineto_nodes_mapping[cnt] = tmp
                cnt += 1
                kineto_nodes[-1].children.append(tmp)
                kineto_nodes.append(tmp)

    # Max call stack depth when building the tree, the deeper the more accurate but takes longer time
    depth = 10

    # Build a tree from the kineto trace
    kineto_graph = transform_to_graph_depth(process_node, depth)
    print("Kineto tree nodes number: ", len(kineto_graph.nodes))

    # Build a tree from the execution trace
    et_graph = transform_to_graph_depth(et_nodes[0], depth)
    print("ET tree nodes number: ", len(et_graph.nodes))

    # print("Kineto nodes info: ", list(kineto_graph.nodes(data=True)))
    # print("ET nodes info: ", list(et_graph.nodes(data=True)))

    # Create the GraphMatcher
    GM = isomorphism.GraphMatcher(kineto_graph, et_graph)

    # Duration of ET nodes
    et_enhanced = {}

    if GM.is_isomorphic():
        mapping = GM.mapping
        print("Graphs are isomorphic")
        for kineto_id, et_id in mapping.items():
            et_enhanced[et_id] = kineto_nodes_mapping[kineto_id].end - kineto_nodes_mapping[kineto_id].start
    else:
        print("Graphs are not isomorphic")

        # # Compute the edit distance using the graph_edit_distance function with node comparison
        # paths, cost = nx.graph_edit_distance(kineto_graph, et_graph, node_compare)
        # print("Tree edit distance:", cost)

        # The problem of finding the exact Graph Edit Distance (GED) is NP-hard so it is often slow
        # and below is a sub-optimal approach

        edit_distance_generator = nx.optimize_graph_edit_distance(kineto_graph, et_graph, node_compare)
        cost = next(edit_distance_generator)

        paths_generator = nx.optimize_edit_paths(kineto_graph, et_graph, node_compare)
        node_edits, edge_edits, cost = next(paths_generator)

        print("Sub-optimal tree edit distance:", cost)

        # print("node_edits: ", node_edits)

        for kineto_id, et_id in node_edits:
            if kineto_id is not None and et_id is not None:
                et_enhanced[et_id] = kineto_nodes_mapping[kineto_id].end - kineto_nodes_mapping[kineto_id].start

    return et_enhanced


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Link kineto trace with execution trace")
    parser.add_argument("--et_file", type=str, required=True, help="Path to the execution trace file")
    parser.add_argument("--kineto_file", type=str, required=True, help="Path to the kineto trace file")
    parser.add_argument("--annotation", default='DataLoader', type=str, help="User annotation of the iteration step of execution trace")
    parser.add_argument("--exact_match", default=False, action='store_true', help="Whether to match the traces exactly")

    args = parser.parse_args()

    # Analysis of execution trace
    with open(args.et_file, 'r') as f:
        et = ExecutionGraph(json.load(f))

    nodes = et.get_nodes(clean=True)

    # Root node of execution trace is 1-based
    et_nodes = collect_nodes(nodes[1])

    print('Number of captured ops in execution trace: ', len(et_nodes))

    # Analysis of kineto trace
    kineto_trace_events = read_dictionary_from_json(args.kineto_file)['traceEvents']

    sorted_kineto_trace_events = sorted(kineto_trace_events, key=lambda kv: kv['ts'])

    kineto_et_events = [event for event in sorted_kineto_trace_events if 'cat' in event and (event['cat'] == 'cpu_op' or event['cat'] == 'user_annotation')]

    kineto_et_segs = []
    kineto_et_seg = []

    # The choice below normally does not matter for approximate match since we rely on the isomorphism of
    # the graphs, but for exact match we will use the execution order and then we should be careful

    # Assume that an iteration ends with the specified annotation
    end_time = -1
    for event in kineto_et_events:
        if end_time > 0 and event['ts'] >= end_time:
            kineto_et_segs.append(kineto_et_seg)
            kineto_et_seg = []
            end_time = -1

        if args.annotation in event['name']:
            kineto_et_seg.append(event)
            end_time = event['ts'] + event['dur']
        else:
            kineto_et_seg.append(event)

    # # Assume that an iteration starts with the specified annotation
    # for event in kineto_et_events:
    #     if args.annotation in event['name']:
    #         kineto_et_segs.append(kineto_et_seg)
    #         kineto_et_seg = [event]
    #     else:
    #         kineto_et_seg.append(event)

    # Find the segment in kineto trace (assuming it contains multiple) with the closest ops to ET 
    # (usually ET has 3 additional annotation ops for processes/threads)
    kineto_et_events = find_closest_segment(kineto_et_segs, len(et_nodes) - 3)

    # # Just a test to randomly choose a segment
    # kineto_et_events = kineto_et_segs[3]

    print('Number of captured ops in kineto trace (should be #ops_in_ET - 3): ', len(kineto_et_events))

    if args.exact_match:
        et_hanced = exact_match(kineto_et_events, et_nodes)
    else:
        et_hanced = approximate_match(kineto_et_events, et_nodes)

    # If linking works, add duration time to each ET node and dump as ET_plus
    if et_hanced:
        with open(args.et_file, 'r') as f:
            et = json.load(f)
            for node in et['nodes']:
                if node['id'] in et_hanced:
                    node['dur'] = et_hanced[node['id']]
        
        et_plus_file = args.et_file.replace('.json', '_plus.json')
        print(f'Enhanced execution trace dumped to {et_plus_file}.')
        with open(et_plus_file, 'w') as f:
            json.dump(et, f, indent=4)