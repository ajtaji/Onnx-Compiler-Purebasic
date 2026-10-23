from pathlib import Path
import onnx,numpy as np,subprocess,re,json,sys
from onnx import helper as h,numpy_helper as nh,TensorProto as T
p=Path(sys.argv[1]).resolve();p.mkdir(parents=True,exist_ok=True);exe=sys.argv[2] if len(sys.argv)>2 else str(Path(__file__).resolve().parents[1]/"bin/PureMetalOnnxCompilerCLI.exe");rows=[]
for variant,target,expected in [('last','pi4',1),('later','pi4',0),('output','pi4',0),('borrowed','pi4',1),('constant','pi4',0),('persistent','pi4',0),('second','pi4',0),('multiple','pi4',1),('last','unoq',1),('last','windows',0),('last','pico',0)]:
 x=h.make_tensor_value_info('X',T.FLOAT,[1,'L',3]);inputs=[x];init=[nh.from_array(np.array(1,np.float32),'one'),nh.from_array(np.ones((1,1,3),np.float32),'tail'),nh.from_array(np.ones((1,2,3),np.float32),'fixed')];nodes=[];outs=[]
 if variant=='borrowed':a='X'
 elif variant=='constant':a='fixed';outs.append(x)
 elif variant=='persistent':a='A';nodes.append(h.make_node('Add',['fixed','one'],['A']))
 else:a='A';nodes.append(h.make_node('Add',['X','one'],['A']))
 concat_inputs=[a,'tail']
 if variant=='persistent':concat_inputs=[a,'X']
 if variant=='second':concat_inputs=['tail',a]
 if variant=='multiple':concat_inputs=[a,'tail','tail']
 nodes.append(h.make_node('Concat',concat_inputs,['Y'],axis=1));outs.append(h.make_tensor_value_info('Y',T.FLOAT,[1,'R',3]))
 if variant=='later':nodes.append(h.make_node('Add',['A','one'],['Z']));outs.append(h.make_tensor_value_info('Z',T.FLOAT,[1,'L',3]))
 if variant=='output':outs.append(h.make_tensor_value_info('A',T.FLOAT,[1,'L',3]))
 m=h.make_model(h.make_graph(nodes,'concat_last_use',inputs,outs,init),opset_imports=[h.make_opsetid('',18)]);m.ir_version=10;source=p/(variant+'.onnx');onnx.save(m,source)
 q=subprocess.run([exe,'--compile',str(source),'--target',target,'--precision','fp32','--output',str(p/(variant+'-'+target))],capture_output=True,encoding='utf8',timeout=60,creationflags=subprocess.CREATE_NO_WINDOW);(p/(variant+'-'+target+'.log')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr
 match=re.search(r'(?:Source:|reusable source emitted to)\s*(.+)',q.stdout);assert match,q.stdout;text=Path(match[1].strip()).read_text();got=len(re.findall(r'^\s*DConcatConsume\(',text,re.M));assert got==expected,(variant,target,got,expected)
 if target in ['pi4','unoq']:
  tensors=int(re.search(r'#PMD_TENSOR_COUNT\s*=\s*(\d+)',text)[1]);reserve=int(re.search(r'#PMD_METADATA_RESERVE\s*=\s*(\d+)',text)[1]);assert reserve==(tensors+2)*17*8+4096
 rows.append(dict(variant=variant,target=target,expected=expected,actual=got))
(p/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
