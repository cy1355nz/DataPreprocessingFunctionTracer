class DataFlowVertex:
    def __init__(self, parent_vertices, name, operation):
        self.parent_vertices = parent_vertices
        self.name = name
        self.operation = operation

    def __repr__(self):
        return "{}, (name={}, op={})".format(self.parent_vertices, self.name, self.operation)


def pipeline_to_dataflow_graph(pipeline):
    graph = []
    layer_graph = []
    def helper(pipeline, name_prefix=[], parent_vertices=[]):
        if 'ColumnTransformer' in str(type(pipeline)):
            for step in pipeline.transformers:
                for column_name in step[2]:
                    helper(step[1], name_prefix+[column_name], parent_vertices)
        elif 'Pipeline' in str(type(pipeline)):
            layer_graph.clear()
            for i, key in enumerate(pipeline.named_steps.keys()):
                helper(pipeline.named_steps[key], name_prefix, parent_vertices+layer_graph)

        else :
            graph.append(DataFlowVertex(parent_vertices, name_prefix[0], str(pipeline).split('(')[0]))
            layer_graph.append(DataFlowVertex(parent_vertices, name_prefix[0], str(pipeline).split('(')[0]))

    helper(pipeline)
    return graph
