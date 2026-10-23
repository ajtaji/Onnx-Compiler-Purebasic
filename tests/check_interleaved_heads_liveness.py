# Adjacent head layout fusion: native64 guards and lifetime boundaries.
from pathlib import Path
import sys,subprocess,re,json,onnx,numpy as np
from onnx import helper as h,numpy_helper as nh,TensorProto as T
p=Path(sys.argv[1]).resolve();p.mkdir(parents=True,exist_ok=True)
exe=str(Path(sys.argv[2]).resolve()) if len(sys.argv)>2 else str(Path(__file__).resolve().parents[1]/'bin/PureMetalOnnxCompilerCLI.exe')
rows=[]
for variant,target,expected in [('last','pi4',1),('laterB','pi4',1),('outputB','pi4',1),('laterT','pi4',0),('outputT','pi4',0),('wrongperm','pi4',0),('separated','pi4',0),('static','pi4',0),('last','unoq',1),('last','windows',0),('last','pico',0)]:
 a=h.make_tensor_value_info('A',T.FLOAT,['B',2,1,4]);b=h.make_tensor_value_info('B',T.FLOAT,['B',4,2,8]);inputs=[a,b];init=[]
 if variant=='static':inputs=[a];init=[nh.from_array(np.ones((1,4,2,8),np.float32),'B')]
 nodes=[h.make_node('Transpose',['B'],['T'],perm=[0,2,1,3] if variant!='wrongperm' else [0,2,3,1])]
 if variant=='separated':nodes.append(h.make_node('Identity',['A'],['AA']))
 nodes.append(h.make_node('MatMul',['AA' if variant=='separated' else 'A','T'],['Y']))
 outs=[h.make_tensor_value_info('Y',T.FLOAT,['B',2,1,8])]
 if variant=='laterB':nodes.append(h.make_node('Identity',['B'],['Z']));outs.append(h.make_tensor_value_info('Z',T.FLOAT,['B',4,2,8]))
 if variant=='outputB':outs.append(b)
 if variant=='laterT':nodes.append(h.make_node('Identity',['T'],['Z']));outs.append(h.make_tensor_value_info('Z',T.FLOAT,['B',2,4,8]))
 if variant=='outputT':outs.append(h.make_tensor_value_info('T',T.FLOAT,['B',2,4,8]))
 m=h.make_model(h.make_graph(nodes,'headlayout',inputs,outs,init),opset_imports=[h.make_opsetid('',18)]);m.ir_version=10
 source=p/(variant+'.onnx');onnx.save(m,source);out=p/(variant+'-'+target)
 q=subprocess.run([exe,'--compile',str(source),'--target',target,'--precision','fp32','--output',str(out)],capture_output=True,encoding='utf8',timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
 (p/(variant+'-'+target+'.log')).write_text(q.stdout+q.stderr);assert q.returncode==0,(variant,target,q.stdout+q.stderr)
 match=re.search(r'(?:Source:|reusable source emitted to)\s*(.+)',q.stdout);assert match,q.stdout
 text=Path(match[1].strip()).read_text();calls=re.findall(r'^\s*DMatMulInterleavedHeads\((.+)\)',text,re.M);assert len(calls)==expected,(variant,target,calls)
 if calls:
  flag=int(calls[0].rsplit(',',1)[1]);assert flag==int(variant=='last'),(variant,flag)
  ids=calls[0].split(',');bid=ids[3];before=text.index('DMatMulInterleavedHeads(')
  # Last-use B must survive until the combined call. The helper releases it.
  assert not re.search(r'^\s*DRelease\('+bid+r'\)\s*$',text[:before],re.M),(variant,'early release')
 rows.append(dict(variant=variant,target=target,expected=expected,actual=len(calls)))
(p/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
