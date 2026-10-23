from pathlib import Path
import onnx,numpy as np,subprocess,json,re,hashlib,sys
from onnx import helper as h,numpy_helper as nh,TensorProto as T
# Usage: python check_int8_fp32_operators.py OUTPUT_DIRECTORY [COMPILER]
r=Path(__file__).resolve().parents[1];p=Path(sys.argv[1]).resolve();p.mkdir(parents=True,exist_ok=True);exe=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else r/'bin/PureMetalOnnxCompilerCLI.exe';results=[]

def run(src,name,ops=None,precision='int8',target='pi4',fail=False):
 out=p/name;args=[str(exe),'--compile',str(src.resolve()),'--target',target,'--precision',precision,'--output',str(out.resolve())]
 if ops is not None:args+=['--int8-fp32-ops',ops]
 q=subprocess.run(args,capture_output=True,encoding='utf8',timeout=60,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));(p/(name+'.log')).write_text(q.stdout+q.stderr,encoding='utf8')
 assert (q.returncode!=0)==fail,(name,q.stdout+q.stderr)
 if fail:assert ('conflicting output-channel layouts' if name.endswith('shared-default') else '--int8-fp32-ops') in q.stdout+q.stderr;results.append(dict(name=name,refused=True));return
 j=json.loads(out.with_suffix('.json').read_text());policy=j.get('fp32_weight_operators',j.get('quantization',{}).get('fp32_weight_operators'));count=j.get('quantized_weight_count',j.get('quantization',{}).get('weight_count'));results.append(dict(name=name,policy=policy,weights=count));return j,count,policy
for dynamic in [False,True]:
 n='N' if dynamic else 1;nodes=[h.make_node('Conv',['X','WC'],['C'],name='convolution'),h.make_node('MatMul',['A','WM'],['Y'],name='linear')];inputs=[h.make_tensor_value_info('X',T.FLOAT,[n,2,12]),h.make_tensor_value_info('A',T.FLOAT,[n,3])];outs=[h.make_tensor_value_info('C',T.FLOAT,[n,4,10]),h.make_tensor_value_info('Y',T.FLOAT,[n,5])]
 init=[nh.from_array((np.arange(24,dtype=np.float32).reshape(4,2,3)-10)/17,'WC'),nh.from_array((np.arange(15,dtype=np.float32).reshape(3,5)-5)/13,'WM')]
 m=h.make_model(h.make_graph(nodes,'independent',inputs,outs,init),opset_imports=[h.make_opsetid('',20)]);m.ir_version=10;src=p/('dynamic.onnx' if dynamic else 'fixed.onnx');onnx.save(m,src);prefix='dynamic-' if dynamic else 'fixed-'
 for name,ops,expected,normalized in [('default',None,2,''),('conv','Conv',1,'Conv'),('matmul','MatMul',1,'MatMul'),('both',' MatMul,conv,Conv ',0,'Conv,MatMul'),('unused','Gemm,LSTM',2,'Gemm,LSTM')]:
  j,count,policy=run(src,prefix+name,ops);assert (count,policy)==(expected,normalized),(name,count,policy)
  assert (j.get('backend')=='reusable-runtime-dimensions')==dynamic
 run(src,prefix+'fp32-reference',precision='fp32')
 assert (p/(prefix+'both.pmw')).read_bytes()==(p/(prefix+'fp32-reference.pmw')).read_bytes()
 assert (p/(prefix+'unused.pmw')).read_bytes()==(p/(prefix+'default.pmw')).read_bytes()
 # Same tensor is a valid Conv and batched MatMul weight. A preserved use
 # must keep it FP32 for every use, while the independent matrix still lowers.
 m.graph.input.append(h.make_tensor_value_info('B',T.FLOAT,[4,1,2]));m.graph.node.append(h.make_node('MatMul',['B','WC'],['Z'],name='shared_linear'));m.graph.output.append(h.make_tensor_value_info('Z',T.FLOAT,[4,1,3]));shared=p/(prefix+'shared.onnx');onnx.save(m,shared)
 run(shared,prefix+'shared-default',fail=True)
 assert run(shared,prefix+'shared-conv','Conv')[1]==1
 assert run(shared,prefix+'shared-matmul','MatMul')[1]==0
 for bad in ['Convv','Conv,','Conv,,MatMul','Softmax']:
  run(src,prefix+'bad-'+str(len(results)),bad,fail=True)
 run(src,prefix+'wrong-precision','Conv',precision='fp32',fail=True)
for dynamic in [False,True]:
 n='N' if dynamic else 1;prefix='dynamic-' if dynamic else 'fixed-'
 for op in ['Gemm','LSTM']:
  if op=='Gemm':
   inputs=[h.make_tensor_value_info('X',T.FLOAT,[n,3])];outs=[h.make_tensor_value_info('Y',T.FLOAT,[n,5])];init=[nh.from_array(np.arange(15,dtype=np.float32).reshape(3,5)/17,'W')];nodes=[h.make_node(op,['X','W'],['Y'])];expected=1
  else:
   inputs=[h.make_tensor_value_info('X',T.FLOAT,[3,n,2])];outs=[h.make_tensor_value_info('Y',T.FLOAT,[3,1,n,3])];init=[nh.from_array(np.arange(24,dtype=np.float32).reshape(1,12,2)/71,'W'),nh.from_array(np.arange(36,dtype=np.float32).reshape(1,12,3)/83,'R')];nodes=[h.make_node(op,['X','W','R'],['Y'],hidden_size=3)];expected=2
  m=h.make_model(h.make_graph(nodes,op,inputs,outs,init),opset_imports=[h.make_opsetid('',20)]);m.ir_version=10;src=p/(prefix+op+'.onnx');onnx.save(m,src)
  assert run(src,prefix+op+'-default')[1]==expected
  assert run(src,prefix+op+'-preserved',op)[1]==0
  assert run(src,prefix+op+'-unaffected','Conv')[1]==expected
  run(src,prefix+op+'-fp32-reference',precision='fp32')
  assert (p/(prefix+op+'-preserved.pmw')).read_bytes()==(p/(prefix+op+'-fp32-reference.pmw')).read_bytes()
(p/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));print('PASS',len(results),'native compiler policy checks')
