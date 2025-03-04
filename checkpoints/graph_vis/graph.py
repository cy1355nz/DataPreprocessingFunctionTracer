class DataFlowVertex:
    def __init__(self, parent_vertices, name, operation):
        self.parent_vertices = parent_vertices
        self.name = name
        self.operation = operation

    def __repr__(self):
        return "{}, (name={}, op={})".format(self.parent_vertices, self.name, self.operation)
# Helper function to topologically sort a DAG 
# HW Support Func
def topo_sort(graph):
    adjacency_list = {vertex.name: [] for vertex in graph}
    visited = {vertex.name: False for vertex in graph}

    for vertex in graph:
        for parent_vertex in vertex.parent_vertices:
            adjacency_list[parent_vertex.name].append(vertex.name)

    output = []

    def toposort(vertex_name, adjacency_list, visited, output):
        visited[vertex_name] = True
        for child_name in adjacency_list[vertex_name]:
            if not visited[child_name]:
                toposort(child_name, adjacency_list, visited, output)
        output.append(vertex_name)

    for vertex_name in adjacency_list.keys():
        if not visited[vertex_name]:
            toposort(vertex_name, adjacency_list, visited, output)

    output.reverse()

    vertices_by_name = {vertex.name: vertex for vertex in graph}

    sorted_graph = []
    for vertex_name in output:
        sorted_graph.append(vertices_by_name[vertex_name])
    return sorted_graph


# Helper function to pick the sink from a single-sink DAG
def find_sink(graph):
    sorted_graph = topo_sort(graph)
    return sorted_graph[-1]


def pipeline_to_dataflow_graph(pipeline):
    graph = []
    layer_graph = []
    # TODO Implement translation of the pipeline into a list of DataFlowVertex objects
    def helper(pipeline, name_prefix=[], parent_vertices=[]):
        if 'ColumnTransformer' in str(type(pipeline)):
            for step in pipeline.transformers:
                for column_name in step[2]:
                    helper(step[1], name_prefix+[step[0]]+[column_name], parent_vertices)
        elif 'Pipeline' in str(type(pipeline)):
            layer_graph.clear()
            for i, key in enumerate(pipeline.named_steps.keys()):
                helper(pipeline.named_steps[key], name_prefix+[key], parent_vertices+layer_graph)

        else :
            graph.append(DataFlowVertex(parent_vertices, '__'.join(name_prefix), str(pipeline).split('(')[0]))
            layer_graph.append(DataFlowVertex(parent_vertices, '__'.join(name_prefix), str(pipeline).split('(')[0]))

    helper(pipeline)
    return graph
