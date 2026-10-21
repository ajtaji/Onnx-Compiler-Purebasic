"""Optional developer tool: emit caller-owned Pi 4 INT8 weight-cache preparation.

Reads generated source metadata only; never evaluates it or opens a weight pack.
Not required to build the compiler or run a generated model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def plan(source):
    if not re.search(r"#PMD_NATIVE_BYTES\s*=\s*8\b", source):
        raise ValueError("Expected a generated 64-bit dynamic model")
    for flag in ('PMO_INT8_BACKEND', 'PMO_USE_INT8'):
        if not re.search(r'#' + flag + r'\s*=\s*1\b', source):
            raise ValueError("Expected the AArch64 INT8 runtime: " + flag)
    size = re.search(r"#PMO_WEIGHT_FILE_BYTES\s*=\s*(\d+)\b", source)
    reset = re.search(r"Procedure PmModelResetRequest\(\)(.*?)\nEndProcedure", source, re.S)
    if not size or not reset or "Procedure.i PmModelBindMemory(" not in source:
        raise ValueError("Missing resident-model metadata")
    size = int(size[1])
    released = set(map(int, re.findall(r"DRelease\((\d+)\)", reset[1])))
    # A complete, contiguous record must identify file-backed storage. Never
    # infer immutable status from a tensor's name or from its current kind.
    records = {}
    pattern = (r"(?m)^  Dt\((\d+)\)\\Data=PmModelWeights\+(\d+)\n"
               r"  Dt\(\1\)\\Kind=(\d+)\n  Dt\(\1\)\\Count=(\d+)\n"
               r"  Dt\(\1\)\\Bytes=(\d+)\n  Dt\(\1\)\\Rank=(\d+)\n")
    for m in re.finditer(pattern, source):
        tid, offset, kind, count, nbytes, rank = map(int, m.groups())
        if kind not in (3, 33):
            continue
        if tid in records or tid in released or not 1 <= rank <= 8:
            raise ValueError("Conflicting immutable tensor metadata: " + str(tid))
        tail = source[m.end():]
        dims = []
        for axis in range(rank):
            d = re.match(r"  DDims\(" + str(tid) + r"\*8\+" + str(axis) + r"\)=(\d+)\n", tail)
            if not d:
                raise ValueError("Missing immutable tensor dimension: " + str(tid))
            dims.append(int(d[1]))
            tail = tail[d.end():]
        if (not all(dims) or math.prod(dims) != count or
                nbytes != count * (2 if kind == 33 else 1) or
                offset > size or nbytes > size - offset):
            raise ValueError("Invalid immutable tensor extent: " + str(tid))
        records[tid] = dict(weight=tid, offset=offset, kind=kind,
                            count=count, bytes=nbytes, dims=dims)
    matrices, conv = [], []
    for tid in sorted(set(map(int, re.findall(r"DMatMul\(\d+,\d+,(\d+)\)", source)))):
        r = records.get(tid)
        if r and len(r['dims']) == 2:
            k, n = r['dims']
            if k*n >= 65536 and n >= 16:
                matrices.append(r)
    calls = (r"DCall\(2\)=(\d+)\nDCall\(3\)=\d+\nDCall\(4\)=\d+\n"
             r"DCall\(5\)=\d+\nDCall\(6\)=\d+\nDCall\(7\)=\d+\nDCall\(8\)=\d+\nDConv\(\)")
    for tid in sorted(set(map(int, re.findall(calls, source)))):
        r = records.get(tid)
        if r and r['kind'] == 3 and len(r['dims']) == 3:
            oc, cg, k = r['dims']
            if cg in (64, 128, 256, 512) and oc >= 4 and oc % 4 == 0 and k > 1:
                conv.append(r)
    if len(matrices) > 64 or len(conv) > 64:
        raise ValueError("Selection exceeds the runtime's 64-entry cache capacity")
    if not matrices and not conv:
        raise ValueError("No eligible immutable weights in the supported emitted format")
    sizes = [sum(r['dims'][0]*(r['dims'][1] & -16)*(2 if r['kind'] == 33 else 1) for r in matrices),
             sum(r['bytes'] for r in conv),
             sum((r['dims'][0]//4)*r['dims'][2]*8 for r in conv)]
    return dict(source_sha256=hashlib.sha256(source.encode('utf8')).hexdigest(),
                weight_bytes=size, matrix=matrices, conv=conv, cache_bytes=sizes)


def emit(p):
    s = ['; Generated immutable-weight cache preparation; include AFTER the model.',
         '; Call only while inference is idle, once after a successful bind.',
         '; Keep source weights and all buffers unchanged. Clear before close/rebind.',
         '; Buffer ranges must also avoid application state, stacks and peripherals.',
         '; Source SHA256: ' + p['source_sha256']]
    for name, size in zip(('MATRIX', 'CONV', 'PROOF'), p['cache_bytes']):
        s.append('#PMO_PREPARED_' + name + '_BYTES=' + str(size))
    s += ['Procedure PmModelClearWeightCaches()',
          ' PmI8PackReset(0,0)',
          ' CompilerIf #PMO_USE_CONV=1',
          ' PmConvPackedCacheReset(0,0) : PmConvFourCacheReset(0,0)',
          ' CompilerEndIf',
          'EndProcedure',
          'Procedure.i PmModelWeightCacheRange(base.i,bytes.i,need.i)',
          ' If need=0 : ProcedureReturn 1 : EndIf',
          ' If base<=0 Or (base & 15)<>0 Or bytes<need Or base>$FFFFFFFF Or bytes>$100000000-base : ProcedureReturn 0 : EndIf',
          ' If base<PmModelWeights+#PMO_WEIGHT_FILE_BYTES And PmModelWeights<base+bytes : ProcedureReturn 0 : EndIf',
          ' If base<DHeap+DHeapBytes And DHeap<base+bytes : ProcedureReturn 0 : EndIf',
          ' ProcedureReturn 1', 'EndProcedure',
          'Procedure.i PmModelPrepareWeightCaches(matrix.i,matrixBytes.i,conv.i,convBytes.i,proof.i,proofBytes.i)',
          ' PmModelClearWeightCaches()',
          ' If PmModelReady=0 Or #PMO_WEIGHT_FILE_BYTES<>' + str(p['weight_bytes']) + ' : ProcedureReturn 0 : EndIf']
    active = []
    for name, cap, macro, need in zip(('matrix', 'conv', 'proof'), ('matrixBytes', 'convBytes', 'proofBytes'), ('MATRIX', 'CONV', 'PROOF'), p['cache_bytes']):
        s.append(f' If PmModelWeightCacheRange({name},{cap},#PMO_PREPARED_{macro}_BYTES)=0 : ProcedureReturn 0 : EndIf')
        if need:
            for other, othercap in active:
                s.append(f' If {name}<{other}+{othercap} And {other}<{name}+{cap} : ProcedureReturn 0 : EndIf')
            active.append((name, cap))
    for r in p['matrix'] + p['conv']:
        tid = r['weight']
        checks = [f'Dt({tid})\\Data<>PmModelWeights+{r["offset"]}',
                  f'Dt({tid})\\Kind<>{r["kind"]}', f'Dt({tid})\\Count<>{r["count"]}',
                  f'Dt({tid})\\Bytes<>{r["bytes"]}', f'Dt({tid})\\Rank<>{len(r["dims"])}']
        checks += [f'DDims({tid}*8+{i})<>{d}' for i, d in enumerate(r['dims'])]
        s.append(' If ' + ' Or '.join(checks) + ' : ProcedureReturn 0 : EndIf')
    s += [' PmI8PackReset(matrix,matrixBytes)', ' CompilerIf #PMO_USE_CONV=1',
          ' PmConvPackedCacheReset(conv,convBytes)',
          ' PmConvFourCacheReset(proof,proofBytes)', ' CompilerEndIf']
    for r in p['matrix']:
        k, n = r['dims']
        s.append(f' If PmI8PackWeight(Dt({r["weight"]})\\Data,{k},{n},{int(r["kind"] == 33)})=0 : PmModelClearWeightCaches() : ProcedureReturn 0 : EndIf')
    for r in p['conv']:
        oc, cg, k = r['dims']
        for api in ('PmConvPackedCacheWeight', 'PmConvFourCacheWeight'):
            s.append(f' If {api}(Dt({r["weight"]})\\Data,{cg},{oc},{k})=0 : PmModelClearWeightCaches() : ProcedureReturn 0 : EndIf')
    return '\n'.join(s + [' ProcedureReturn 1', 'EndProcedure', ''])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    p = plan(args.model.read_text(encoding='utf8'))
    if args.output.exists() or args.output.with_suffix('.json').exists():
        ap.error('Output exists; choose a fresh destination')
    args.output.write_text(emit(p), encoding='utf8')
    args.output.with_suffix('.json').write_text(json.dumps(p, indent=2) + '\n', encoding='utf8')
    print(f'{len(p["matrix"])} matrices, {len(p["conv"])} convolutions; cache bytes {p["cache_bytes"]}')


if __name__ == '__main__':
    main()
