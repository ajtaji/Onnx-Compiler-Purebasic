"""A numeric fixture and valid unsupported-layout refusals for KV IO export."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh, TensorProto as T

spec = importlib.util.spec_from_file_location('split_kv', Path(__file__).with_name('split_recurrent_kv.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    initializers = [nh.from_array(np.array(0, np.int64), 'zero'),
                    nh.from_array(np.array(1, np.int64), 'one_index'),
                    nh.from_array(np.array([0], np.int64), 'axes'),
                    nh.from_array(np.array(1, np.float32), 'one')]
    nodes = [h.make_node('Gather', ['cache', 'zero'], ['old_key'], axis=0),
             h.make_node('Gather', ['cache', 'one_index'], ['old_value'], axis=0),
             h.make_node('Add', ['old_key', 'one'], ['key']),
             h.make_node('Add', ['old_value', 'one'], ['value']),
             h.make_node('Unsqueeze', ['key', 'axes'], ['stack_key']),
             h.make_node('Unsqueeze', ['value', 'axes'], ['stack_value']),
             h.make_node('Concat', ['stack_key', 'stack_value'], ['next'], axis=0)]
    graph = h.make_graph(nodes, 'paired_state',
                         [h.make_tensor_value_info('cache', T.FLOAT, [2, 1, 3])],
                         [h.make_tensor_value_info('next', T.FLOAT, [2, 1, 3])], initializers)
    result = h.make_model(graph, opset_imports=[h.make_opsetid('', 18)])
    result.ir_version = 10
    return result


model = fixture()
result, report = module.split_recurrent_kv(model, [('cache', 'next')])
assert report['removed_nodes'] == 5 and len(result.graph.node) == 2
assert [x.SerializeToString() for x in model.graph.initializer] == [x.SerializeToString() for x in result.graph.initializer]
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.log_severity_level = 3
a = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
before = ort.InferenceSession(model.SerializeToString(), sess_options=options).run(None, {'cache': a})[0]
after = ort.InferenceSession(result.SerializeToString(), sess_options=options).run(None, {'cache_key': a[0], 'cache_value': a[1]})
assert before.tobytes() == np.stack(after).tobytes()
rows = [dict(case='numeric_and_initializer_identity', passed=True)]
for case in ['missing_input', 'missing_output', 'duplicate_pair', 'extra_reader',
             'leading_extent', 'nonleading_gather', 'unsqueeze_axis', 'extra_output', 'name_collision']:
    candidate = fixture()
    pairs = [('cache', 'next')]
    if case == 'missing_input':
        pairs = [('absent', 'next')]
    elif case == 'missing_output':
        pairs = [('cache', 'absent')]
    elif case == 'duplicate_pair':
        pairs *= 2
    elif case == 'extra_reader':
        candidate.graph.node.append(h.make_node('Identity', ['cache'], ['extra']))
    elif case == 'leading_extent':
        candidate.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 3
    elif case == 'nonleading_gather':
        candidate.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 2
        candidate.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2
        for node in candidate.graph.node[:2]:
            node.attribute[0].i = 1
    elif case == 'unsqueeze_axis':
        candidate.graph.initializer[2].CopyFrom(nh.from_array(np.array([1], np.int64), 'axes'))
    elif case == 'extra_output':
        candidate.graph.output.append(h.make_tensor_value_info('key', T.FLOAT, [1, 3]))
    elif case == 'name_collision':
        candidate.graph.input.append(h.make_tensor_value_info('cache_key', T.FLOAT, [1, 3]))
    onnx.checker.check_model(candidate)
    try:
        module.split_recurrent_kv(candidate, pairs)
    except ValueError:
        rows.append(dict(case=case, passed=True))
    else:
        raise AssertionError('Unsupported graph silently accepted: ' + case)
print(json.dumps(rows, indent=2))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=2), encoding='utf8')
