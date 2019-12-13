from collections import defaultdict
import inspect
import pandas as pd
import numpy as np
from scipy import stats
import re
np.set_printoptions(precision = 4)
pd.set_option("display.precision", 4)
class DataFlowVertex:
    def __init__(self, parent_vertices, name, operation, params):
        self.parent_vertices = parent_vertices
        self.name = name
        self.operation = operation
        self.params = params
        
    def __repr__(self):
        return "(vertex_name={}: parent={}, op={}, params={})".format(self.name, self.parent_vertices, self.operation, self.params)
    
    def display(self):
        print("(vertex_name={}: parent={}, op={}, params={})".format(self.name, self.parent_vertices, self.operation, self.params))

    
def pipeline_to_dataflow_graph(pipeline):
    '''
    Support function used in Logs. Parent_vertices are all set to None here. 
    name field is set to be column name in ColumnTransfomer and operation field contains not only name but also input args.
    '''
    graph = []
    layer_graph = []
    def helper(pipeline, name_prefix=[], parent_vertices=[]):
        if 'ColumnTransformer' in str(type(pipeline)):
            for step in pipeline.transformers:
                for column_name in step[2]:
                    helper(step[1], name_prefix+[column_name], parent_vertices)
        elif 'Pipeline' in str(type(pipeline)):
            for i, key in enumerate(pipeline.named_steps.keys()):
                helper(pipeline.named_steps[key], name_prefix, parent_vertices)

        else :
            graph.append(DataFlowVertex(parent_vertices, ''.join(name_prefix), pipeline, None))

    helper(pipeline)
    return graph

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


def find_sink(graph):
    sorted_graph = topo_sort(graph)
    return sorted_graph[-1]
        
def search_vertex_by_names(names, vertices_list):
    result = []
    names_to_drop = set()
    for vertex in set(vertices_list):
        if vertex.name.split('_')[0] in names:
            if '_' in vertex.name:
                names_to_drop.add(vertex.name.split('_')[0])
            result.append(vertex)
    for item in result:
        if item.name in names_to_drop:
            result.remove(item)
    return result

def func_aggregation(func_str):
    '''
    This function is used for line execution with exec()

    args:
        function strings after inspect
    returns:
        list of functionable strings for exec()
    '''

    res = [] # executables for return
    stack_for_parent = [] # stack storing brackets for line integration
    logs_of_parent = [] # logs of lines for concat
#     convert function codes to list of strings
    func_list = [item.strip() for item in func_str.split('\n')]
    i = 0
    while True:
        if func_list[i].startswith('def'):
            func_list = func_list[i:]
            break
        i = i + 1
#     function args
    func_args = [item.strip() for item in func_list[0].split('(')[1].rstrip('):').split(',')]
    for item in func_list[1:]:
        if not item:
            continue
        logs_of_parent.append(item)
        for char in item:
            if char == '(':
                stack_for_parent.append('(')
            if char == '[':
                stack_for_parent.append('[')
            if char == ')' and stack_for_parent[-1] == '(':
                stack_for_parent.pop(-1)
            if char == ']' and stack_for_parent[-1] == '[':
                stack_for_parent.pop(-1)
        if not stack_for_parent:
            res.append(''.join(logs_of_parent))
            logs_of_parent.clear()
    return func_args, res[:-1], [item.strip() for item in res[-1].replace('return ', '').split(',')]

def handle_dict(dict_1, dict_2):
    '''
    Calculate differences between two dictionaries
    eg: input: d1 = {'a': 1, 'b': 2, 'c': 3, 'e': 2, 'f':4}
               d2 = {'a': 10, 'b': 9, 'c': 8, 'd': 7, 'e': 3}
        output: z = {'a': -9, 'b': -7, 'c': -5, 'e': -1, 'f': 4, 'd': -7}
    '''

    dict_1_re = {key: round(dict_1.get(key,0) - dict_2.get(key, 0), 4) for key in dict_1.keys()}
    dict_2_re = {key: round(dict_1.get(key,0) - dict_2.get(key, 0), 4) for key in dict_1.keys()}
    return {**dict_1_re, **dict_2_re}

def get_categorical_dif(cat_df, cat_metric_list, prev):
    '''
    Calculate differences for categorical dataframe comparison
    Need special handling for 'class_count' and 'class_percent' 
        since they are stored as dict in the dataframe
    '''
    cat_dif = pd.DataFrame()
    for i in cat_metric_list:
        if i != 'class_count' and i != 'class_percent':
            # if the metric is not defined as a dictionary
            dif = cat_df[i] - prev[i]
            cat_dif[i] = dif
        else:
            for idx, col in enumerate(cat_df.index):
                dif = handle_dict(cat_df[i][idx], prev[i][idx])
                cat_dif.loc[col, i] = [dif]
    return cat_dif

def cal_numerical(target_df_1, numeric_feature, numerical_df):
    '''
    Calculate metrices for numerical features
        including counts, missing values, Median and MAD, range/scaling
    '''

    # get counts of non NA values
    count_log = target_df_1[numeric_feature].count()
    numerical_df.loc[numeric_feature, 'count'] = count_log

    # get missing value counts
    missing_count_log = target_df_1[numeric_feature].isna().sum()
    numerical_df.loc[numeric_feature, 'missing_count'] = missing_count_log

    # distribution
    # Median and MAD
    median_log = target_df_1[numeric_feature].median()
    numerical_df.loc[numeric_feature, 'median'] = median_log
    if missing_count_log == 0:
        mad_log = stats.median_absolute_deviation(target_df_1[numeric_feature])
        numerical_df.loc[numeric_feature, 'mad'] = mad_log
    else:
        numerical_df.loc[numeric_feature, 'mad'] = 0

    # range/ scaling
    range_log = target_df_1[numeric_feature].max() - target_df_1[numeric_feature].min()
    numerical_df.loc[numeric_feature, 'range'] = range_log

    return numerical_df  

def cal_categorical(target_df_1, cat_feature, cat_df):
    '''
    Calculate metrices for categorical features
        including missing values, number of classes, counts for each group, percentage for each group
    '''

    # get missing value counts
    missing_count_log = target_df_1[cat_feature].isna().sum()
    cat_df.loc[cat_feature, 'missing_count'] = missing_count_log

    # get number of classes
    num_class_log = len(target_df_1[cat_feature].value_counts().keys())
    cat_df.loc[cat_feature, 'num_class'] = num_class_log

    # get counts for each group
    class_count_log = target_df_1[cat_feature].value_counts().to_dict()
    cat_df.loc[cat_feature, 'class_count'] = [class_count_log]

    # get percentage each group covers
    class_percent_log = (target_df_1[cat_feature].value_counts() / \
    target_df_1[cat_feature].value_counts().sum()).to_dict()
    cat_df.loc[cat_feature, 'class_percent'] = [class_percent_log]

    return cat_df 

def pipeline_to_dataflow_graph_full(pipeline, name_prefix='', parent_vertices=[]):
    graph = []
    parent_vertices_for_current_step = parent_vertices
    parent_vertices_for_next_step = []

    for step_name, component in pipeline.steps:
        component_class_name = component.__class__.__name__

        if component_class_name == 'ColumnTransformer':
            for transformer_prefix, transformer_component, columns in component.transformers:
                for column in columns:
                    name = column
                    transformer_component_class_name = transformer_component.__class__.__name__

                    if transformer_component_class_name == 'Pipeline':

                        vertices_to_add = pipeline_to_dataflow_graph_full(transformer_component,
                                                                     name + "__",
                                                                     parent_vertices_for_current_step)
                        
                        for vertex in vertices_to_add:
                            graph.append(vertex)

                        parent_vertices_for_next_step.append(find_sink(vertices_to_add))

                    else:
                        vertex = DataFlowVertex(parent_vertices_for_current_step,
                                                name_prefix + name,
                                                transformer_component_class_name,
                                               '')
                        graph.append(vertex)
                        parent_vertices_for_next_step.append(vertex)

        else:
            vertex = DataFlowVertex(parent_vertices_for_current_step,
                                    name_prefix + step_name,
                                    component_class_name,
                                   '')
            graph.append(vertex)
            parent_vertices_for_next_step.append(vertex)

        parent_vertices_for_current_step = parent_vertices_for_next_step.copy()
        parent_vertices_for_next_step = []

    return graph

def check_cat_dif(cat_dif_):
        
    for i in cat_dif_.columns:
        if i != 'class_count' and i != 'class_percent':
            if cat_dif_.loc[:,i].any() !=0:
                return True  # dif exists
        else:
            for sub_dict in cat_dif_.loc[:,i].values:
                if sum(sub_dict.values()) != 0:
                    return True # dif exists
    return False # cat_dif is all zero