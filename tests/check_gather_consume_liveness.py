# Check last-use Gather rewriting, retained values, constants and native64 guards.
# Usage: python check_gather_consume_liveness.py OUTPUT_DIRECTORY [COMPILER]
from pathlib import Path
import onnx,numpy as np,subprocess,re,json,sys
from onnx import helper as h,numpy_helper as nh,TensorProto as T
p=Path(sys.argv[1]).resolve();p.mkdir(parents=True,exist_ok=True);exe=str(Path(sys.argv[2]).resolve()) if len(sys.argv)>2 else str(Path(__file__).resolve().parents[1]/'bin/PureMetalOnnxCompilerCLI.exe');rows=[]
for variant,target,expected in [('last','pi4',1),('later','pi4',0),('output','pi4',0),('borrowed','pi4',1),('constant','pi4',0),('last','unoq',1),('last','windows',0),('last','pico',0)]:
 x=h.make_tensor_value_info('X',T.FLOAT,['N',2,3]);init=[nh.from_array(np.array(1,dtype=np.float32),'one'),nh.from_array(np.array(0,dtype=np.int64),'index')];nodes=[];inputs=[x];outs=[]
 if variant=='borrowed':a='X'
 elif variant=='constant':a='A';init.append(nh.from_array(np.arange(6,dtype=np.float32).reshape(2,1,3),'A'));inputs=[x];outs.append(h.make_tensor_value_info('X',T.FLOAT,['N',2,3]))
 else:a='A';nodes.append(h.make_node('Add',['X','one'],['A']))
 nodes.append(h.make_node('Gather',[a,'index'],['Y'],axis=0));outs.append(h.make_tensor_value_info('Y',T.FLOAT,[1,3] if variant=='constant' else [2,3]))
 if variant=='later':nodes.append(h.make_node('Add',['A','one'],['Z']));outs.append(h.make_tensor_value_info('Z',T.FLOAT,['N',2,3]))
 if variant=='output':outs.append(h.make_tensor_value_info('A',T.FLOAT,['N',2,3]))
 m=h.make_model(h.make_graph(nodes,'liveness',inputs,outs,init),opset_imports=[h.make_opsetid('',18)]);m.ir_version=10;source=p/(variant+'.onnx');onnx.save(m,source);out=p/(variant+'-'+target)
 q=subprocess.run([exe,'--compile',str(source.resolve()),'--target',target,'--precision','fp32','--output',str(out.resolve())],capture_output=True,encoding='utf8',timeout=60,creationflags=subprocess.CREATE_NO_WINDOW);(p/(variant+'-'+target+'.log')).write_text(q.stdout+q.stderr);assert q.returncode==0,(variant,target,q.stdout+q.stderr)
 suffix='.pb' if target=='windows' else '.pi4' if target=='pi4' else '.unoq' if target=='unoq' else '.pico'
 # CLI log provides the exact emitted source path; do not search folders.
 match=re.search(r'Source:\s*(.+)',q.stdout);path=Path(match[1].strip()) if match else out.with_suffix(suffix)
 if not path.is_file():
  paths=re.findall(r'([A-Z]:[^\r\n]+\.(?:pb|pi4|unoq|pico))',q.stdout);assert paths,q.stdout;path=Path(paths[0].strip())
 text=path.read_text();assert '#PMD_TENSOR_COUNT' in text,(variant,target,'Fixture did not select runtime-dimension emitter')
 if target in ['pi4','unoq']:
  tensors=int(re.search(r'#PMD_TENSOR_COUNT\s*=\s*(\d+)',text)[1]);reserve=int(re.search(r'#PMD_METADATA_RESERVE\s*=\s*(\d+)',text)[1]);assert reserve==(tensors+2)*17*8+4096
 got=len(re.findall(r'^\s*DGatherConsume\(',text,re.M));assert got==expected,(variant,target,got,expected)
 rows.append(dict(variant=variant,target=target,expected=expected,actual=got,source=str(path)))
print(json.dumps(rows,indent=2));(p/'results.json').write_text(json.dumps(rows,indent=2))
