"""Checked structural export of explicitly paired stacked recurrent KV states.

Split only a leading pair read by scalar Gather(0/1) and rebuilt by an
axis-zero Concat of two Unsqueeze(0) nodes. Arithmetic and initializers stay
unchanged. Applications must adopt the reported input/output interface.
"""
from pathlib import Path
import argparse
import copy
import json
import onnx
from onnx import numpy_helper


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def split_recurrent_kv(model, pairs):
    result = copy.deepcopy(model)
    graph = result.graph
    nodes = list(graph.node)
    inputs = {value.name: value for value in graph.input}
    outputs = {value.name: value for value in graph.output}
    consumers = {}
    producers = {}
    constants = {value.name: numpy_helper.to_array(value) for value in graph.initializer}
    names = set(inputs) | set(outputs)
    for index, node in enumerate(nodes):
        for value in node.input:
            consumers.setdefault(value, []).append((index, node))
            names.add(value)
        for value in node.output:
            producers[value] = (index, node)
            names.add(value)
        if node.op_type == 'Constant':
            value = next((a for a in node.attribute if a.name == 'value'), None)
            if value is not None:
                constants[node.output[0]] = numpy_helper.to_array(value.t)
    pair_map = dict(pairs)
    _require(len(pair_map) == len(pairs), 'Duplicate recurrent input pair.')
    _require(len(set(pair_map.values())) == len(pairs), 'Duplicate recurrent output pair.')
    remove = set()
    rename = {}
    replacements = {}
    output_replacements = {}
    report = []
    for input_index, original in enumerate(graph.input):
        if original.name not in pair_map:
            continue
        output_name = pair_map[original.name]
        _require(output_name in outputs, 'Paired graph output does not exist: ' + output_name)
        outgoing = outputs[output_name]
        for value in (original, outgoing):
            dims = value.type.tensor_type.shape.dim
            _require(dims and dims[0].HasField('dim_value') and dims[0].dim_value == 2,
                     'Recurrent state must declare a leading extent of two: ' + value.name)
        _require(original.type.tensor_type.elem_type == outgoing.type.tensor_type.elem_type,
                 'Paired state element types differ.')
        _require(len(original.type.tensor_type.shape.dim) == len(outgoing.type.tensor_type.shape.dim),
                 'Paired state ranks differ.')
        readers = {}
        for index, node in consumers.get(original.name, []):
            _require(node.op_type == 'Gather' and len(node.input) == 2 and node.input[0] == original.name,
                     'Stacked input has an unsupported consumer: ' + original.name)
            axis = next((a.i for a in node.attribute if a.name == 'axis'), 0)
            selector = constants.get(node.input[1])
            _require(axis == 0 and selector is not None and selector.ndim == 0 and int(selector) in (0, 1),
                     'State Gather must select scalar zero or one on axis zero.')
            part = int(selector)
            _require(part not in readers, 'More than one Gather reads a state part.')
            readers[part] = node.output[0]
            remove.add(index)
        _require(set(readers) == {0, 1}, 'Both state parts must have one Gather reader.')
        _require(output_name in producers, 'Stacked output has no producer.')
        index, concat = producers[output_name]
        axis = next((a.i for a in concat.attribute if a.name == 'axis'), None)
        _require(concat.op_type == 'Concat' and axis == 0 and len(concat.input) == 2,
                 'Paired output must stack exactly two parts on axis zero.')
        remove.add(index)
        sources = []
        for name in concat.input:
            _require(name in producers, 'Stacked output part has no producer.')
            index, unsqueeze = producers[name]
            _require(unsqueeze.op_type == 'Unsqueeze' and len(unsqueeze.input) == 2
                     and len(consumers.get(name, [])) == 1,
                     'Stacked output needs single-use Unsqueeze parts.')
            axes = constants.get(unsqueeze.input[1])
            _require(axes is not None and axes.size == 1 and int(axes.reshape(-1)[0]) == 0,
                     'Output Unsqueeze must insert axis zero.')
            source = unsqueeze.input[0]
            _require(source not in inputs and source not in outputs and source not in rename
                     and source not in readers.values(),
                     'Output part aliases an existing interface or renamed part.')
            sources.append(source)
            remove.add(index)
        new_inputs = []
        new_outputs = []
        for part, label in enumerate(('key', 'value')):
            incoming = copy.deepcopy(original)
            incoming.name += '_' + label
            outgoing_part = copy.deepcopy(outgoing)
            outgoing_part.name += '_' + label
            _require(incoming.name not in names and outgoing_part.name not in names,
                     'A split state name already exists.')
            names.update((incoming.name, outgoing_part.name))
            del incoming.type.tensor_type.shape.dim[0]
            del outgoing_part.type.tensor_type.shape.dim[0]
            rename[readers[part]] = incoming.name
            rename[sources[part]] = outgoing_part.name
            new_inputs.append(incoming)
            new_outputs.append(outgoing_part)
        replacements[original.name] = new_inputs
        output_replacements[output_name] = new_outputs
        report.append(dict(original_input=original.name, original_output=output_name,
                           original_input_index=input_index,
                           inputs=[x.name for x in new_inputs], outputs=[x.name for x in new_outputs]))
    _require(len(report) == len(pair_map), 'A paired graph input does not exist.')
    rewritten = []
    for index, node in enumerate(nodes):
        if index in remove:
            continue
        node = copy.deepcopy(node)
        for i, value in enumerate(node.input):
            node.input[i] = rename.get(value, value)
        for i, value in enumerate(node.output):
            node.output[i] = rename.get(value, value)
        rewritten.append(node)
    new_inputs = []
    for value in graph.input:
        new_inputs.extend(replacements.get(value.name, [copy.deepcopy(value)]))
    new_outputs = []
    for value in graph.output:
        new_outputs.extend(output_replacements.get(value.name, [copy.deepcopy(value)]))
    removed_values = {value for index in remove for value in nodes[index].output} | set(replacements)
    value_info = []
    for value in graph.value_info:
        if value.name in removed_values:
            continue
        value = copy.deepcopy(value)
        value.name = rename.get(value.name, value.name)
        value_info.append(value)
    del graph.node[:]
    graph.node.extend(rewritten)
    del graph.input[:]
    graph.input.extend(new_inputs)
    del graph.output[:]
    graph.output.extend(new_outputs)
    del graph.value_info[:]
    graph.value_info.extend(value_info)
    onnx.checker.check_model(result)
    return result, dict(pairs=report, removed_nodes=len(remove), original_nodes=len(nodes),
                        nodes=len(rewritten), inputs=len(new_inputs), outputs=len(new_outputs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--pair', nargs=2, action='append', required=True, metavar=('INPUT', 'OUTPUT'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    _require(args.source.resolve() != args.destination.resolve(), 'Source and destination must differ.')
    result, report = split_recurrent_kv(onnx.load(str(args.source)), args.pair)
    onnx.save(result, str(args.destination))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
