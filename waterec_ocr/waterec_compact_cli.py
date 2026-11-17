"""Experimental compact OCR workflow; dedicated prototype, standard CLI unchanged.

Use ``python -m waterec_ocr.waterec_compact_cli --help``. No weights are
bundled. TensorRT/ONNX use explicit fixed-capacity deployments; native Torch
can batch images. Omit --static-batch to retain the original B1 export. Model,
dataset and deployment accuracy must be qualified separately for each task.
SPDX-License-Identifier: Apache-2.0
"""
import argparse,hashlib,json,os,resource,threading,time
from pathlib import Path
from .data_io import IMAGE_EXTENSIONS,scan_pairs
from .metrics import score_pairs,score_price_values


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def _config(path):
    import yaml
    path=Path(path).expanduser().resolve()
    if not path.is_file():raise FileNotFoundError(path)
    config=yaml.safe_load(path.read_text())
    if not isinstance(config,dict):raise ValueError('Config must be a YAML mapping')
    for key in ['dataset_root','output_dir','pretrained_dir','teacher_checkpoint','validation_folder']:
        if config.get(key) is not None:
            p=Path(config[key]).expanduser();config[key]=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    return config

def _metadata(folder,checkpoint=None):
    folder=Path(folder);path=folder/'metadata.json'
    if not path.is_file():raise FileNotFoundError(path)
    m=json.loads(path.read_text());a=m.get('compact_architecture',{})
    if m.get('family')=='waterec_compact_batch':
        from .waterec_compact_batch_assets import load_metadata
        m=load_metadata(folder)
        if checkpoint is not None:
            if not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
            if _sha(checkpoint)!=m['checkpoint_sha256']:raise ValueError('Requested checkpoint SHA differs from deployment; export/build the matching typed checkpoint')
        return m
    if m.get('family')!='waterec_compact' or m.get('schema')!=1 or a.get('family')!='waterec_compact' or a.get('architecture')!='rank_mlp_image_topk':raise ValueError('Explicit typed compact metadata required; standard R/RS is not compact')
    if m.get('static_batch')!=1 or m.get('max_len')!=25 or type(m.get('max_tokens')) is not int or m['max_tokens']<64 or not 1<=m.get('memory_tokens',0)<=m['max_tokens']:raise ValueError('Unsupported compact static B1 deployment')
    if checkpoint is not None:
        if not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
        if not m.get('checkpoint_sha256') or _sha(checkpoint)!=m['checkpoint_sha256']:raise ValueError('Requested checkpoint SHA differs from deployment; export/build the matching typed checkpoint')
    return m

class _TorchRecognizer:
    def __init__(self,checkpoint,device):
        import torch
        from .waterec_compact_models import load_compact_checkpoint
        self.torch=torch;self.device=device;self.model=load_compact_checkpoint(checkpoint,device=device);n=self.model.native
        self.meta={'normalization':self.model.config.get('normalization','none'),'task':self.model.config['task'],'characters':n.characters,'bos':n.decoder.bos,'max_len':25,'max_tokens':self.model.config['tokens']}
    def infer(self,images):
        with self.torch.inference_mode():return self.model.decode(self.model([self.model.preprocess(image).to(self.device) for image in images]))
    def close(self):self.model=None

def _recognizer(args):
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    if args.backend!='torch' and args.batch_size!=1 and (args.deployment is None or not (args.deployment/'metadata.json').is_file()):raise ValueError('Compact ONNX/TensorRT use static B1 unless an explicit fixed-batch deployment is supplied')
    if args.backend=='onnx' and args.device!='cpu':raise ValueError('Compact ONNX runtime currently uses CPU; --device applies to Torch')
    if args.backend=='torch':
        if args.checkpoint is None:raise ValueError('Native compact inference requires --checkpoint with explicit typed metadata')
        if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
        return _TorchRecognizer(args.checkpoint,args.device)
    if args.deployment is None:raise ValueError('ONNX/TensorRT require --deployment')
    metadata=_metadata(args.deployment,args.checkpoint)
    if metadata['family']=='waterec_compact_batch':
        if args.batch_size>metadata['static_batch']:raise ValueError('batch-size exceeds the fixed deployment capacity')
        if args.backend=='onnx':
            from .waterec_compact_batch_onnx import BatchONNXRecognizer
            return BatchONNXRecognizer(args.deployment)
        from .waterec_compact_batch_runtime import BatchCompactRecognizer
        return BatchCompactRecognizer(args.deployment)
    if args.batch_size!=1:raise ValueError('Compact ONNX/TensorRT static B1 deployment requires batch-size1')
    if args.backend=='tensorrt' and metadata['max_tokens']>2048:raise ValueError('Compact FP16 TensorRT budget must be64..2048; native/CPU ONNX budgets are separate')
    if args.backend=='onnx':
        from .waterec_compact_onnx import ONNXRecognizer
        return ONNXRecognizer(args.deployment)
    from .waterec_compact_runtime import CompactRecognizer
    return CompactRecognizer(args.deployment)

def _image(path):
    from PIL import Image
    with Image.open(path) as image:return image.convert('RGB')

def _rows(args,metadata):
    if args.command=='evaluate':
        rows=scan_pairs(args.folder,metadata.get('normalization','none'))
        allowed=set(metadata['characters'][1:metadata['bos']])
        for row in rows:
            if len(row['label'])>24 or any(c not in allowed for c in row['label']):raise ValueError('Unsupported/max24 label: '+row['image'])
        return rows
    if getattr(args,'image',None) is not None:
        if not args.image.is_file():raise FileNotFoundError(args.image)
        return [{'image':args.image.name,'path':str(args.image)}]
    if args.folder is None or not args.folder.is_dir():raise ValueError('Supply an existing --image or --folder')
    rows=[{'image':p.name,'path':str(p)} for p in sorted(args.folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
    if not rows:raise ValueError('No images found')
    return rows

def _rss():
    path=Path('/proc/self/status')
    if not path.exists():return None
    return int(next(line.split()[1] for line in path.read_text().splitlines() if line.startswith('VmRSS:')))*1024

class _MemorySampler:
    """Optional sampled driver accounting, not minimum RAM/instantaneous peak."""
    def __init__(self):
        self.stop=threading.Event();self.peak=0;self.samples=0;self.error=None;self.thread=None
        try:
            import pynvml as nv
            nv.nvmlInit();physical=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0];self.nv=nv;self.handle=nv.nvmlDeviceGetHandleByUUID(physical) if physical.startswith('GPU-') else nv.nvmlDeviceGetHandleByIndex(int(physical));self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
        except Exception as error:self.error=repr(error)
    def _run(self):
        while not self.stop.is_set():
            try:
                values=[int(p.usedGpuMemory) for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle) if p.pid==os.getpid() and 0<=int(p.usedGpuMemory)<1<<50];self.peak=max(self.peak,sum(values));self.samples+=1
            except Exception as error:self.error=repr(error);return
            self.stop.wait(.002)
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join()
        return {'nvml_process_peak_bytes':self.peak,'samples':self.samples,'interval_seconds':.002,'error':self.error,'scope':'Sampled driver process usage including context; not minimumRAM or allocation trace'}

def _predict(args):
    benchmark=args.command=='benchmark'
    if getattr(args,'image',None) is not None and not args.image.is_file():raise FileNotFoundError(args.image)
    if getattr(args,'folder',None) is not None and not args.folder.is_dir():raise NotADirectoryError(args.folder)
    if benchmark and args.repeats<1:raise ValueError('repeats must be positive')
    before=_rss();sampler=_MemorySampler() if benchmark and (args.backend=='tensorrt' or args.device.startswith('cuda')) else None;model=None;cache={}
    try:
        model=_recognizer(args);after=_rss();metadata=model.meta;rows=_rows(args,metadata);fixed_batch=metadata.get('static_batch',1) if metadata.get('family')=='waterec_compact_batch' else None
        if benchmark and args.predecoded:cache={r['path']:_image(r['path']) for r in rows}
        if benchmark:
            for i in range(5):
                if fixed_batch:
                    warm=rows[:fixed_batch];images=[cache.get(row['path']) or _image(row['path']) for row in warm]
                    try:model.infer_batch(images)
                    finally:
                        for row,image in zip(warm,images):
                            if row['path'] not in cache:image.close()
                else:
                    row=rows[i%len(rows)];image=cache.get(row['path']) or _image(row['path'])
                    try:model.infer([image])
                    finally:
                        if row['path'] not in cache:image.close()
        ready=_rss();records=[];timings=[]
        for repeat in range(args.repeats if benchmark else 1):
            for off in range(0,len(rows),args.batch_size):
                batch=rows[off:off+args.batch_size];start=time.perf_counter_ns();images=[cache.get(r['path']) or _image(r['path']) for r in batch]
                try:predictions=model.infer_batch(images) if fixed_batch else model.infer(images)
                finally:
                    for row,image in zip(batch,images):
                        if row['path'] not in cache:image.close()
                elapsed=(time.perf_counter_ns()-start)/1e6
                if len(predictions)!=len(batch):raise ValueError('Recognizer output batch mismatch')
                timing={'repeat':repeat,'real_images':len(batch),'batch_ms':elapsed}
                if fixed_batch:timing['physical_images']=fixed_batch
                timings.append(timing)
                for row,text in zip(batch,predictions):
                    record={'image':row['image'],'prediction':text,'repeat':repeat}
                    if 'label' in row:record['gt']=row['label']
                    records.append(record)
        result={'backend':args.backend,'records':records,'images':len(rows),'batch_size':args.batch_size,'static_deployment_B1':args.backend!='torch' and fixed_batch in [None,1],'scope':'Image-to-rawtext only; no correction, orientation or speculative inference'}
        if fixed_batch:result['static_deployment_batch']=fixed_batch
        if args.command=='evaluate':
            labels=[r['label'] for r in rows];preds=[r['prediction'] for r in records];result['metrics']=score_pairs(labels,preds,metadata.get('normalization','none'))
            if metadata.get('task')=='price':result['metrics'].update(score_price_values(labels,preds))
        if benchmark:
            import numpy as np
            first={r['image']:r['prediction'] for r in records if r['repeat']==0};result.update(repeats=args.repeats,warmups=5,timed_images=len(rows)*args.repeats,consistent_raw_across_passes=all(first[r['image']]==r['prediction'] for r in records),timed_batches=timings,weighted_ms_per_image=sum(r['batch_ms'] for r in timings)/(len(rows)*args.repeats),images_per_second=len(rows)*args.repeats*1000/sum(r['batch_ms'] for r in timings),batch_ms={'mean':float(np.mean([r['batch_ms'] for r in timings])),'median':float(np.median([r['batch_ms'] for r in timings])),'p95':float(np.percentile([r['batch_ms'] for r in timings],95))},predecoded=args.predecoded,host_bytes={'before_load':before,'after_load':after,'ready':ready,'after_run':_rss(),'ru_maxrss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},PID=os.getpid(),memory=sampler.close() if sampler else None,timing_scope='Completed full AR andUnicode; PIL filedecode included unlesspredecoded; load/setup/5warmups excluded timing; memory coversload/warmup/run. CurrentRSS vs highwatermark notminimumRAM.')
            sampler=None
            if fixed_batch:
                full=[r['batch_ms'] for r in timings if r['real_images']==fixed_batch]
                result.update(physical_timed_slots=sum(r['physical_images'] for r in timings),partial_batches=[r for r in timings if r['real_images']<fixed_batch],full_batch_ms={'count':len(full),'mean':float(np.mean(full)),'median':float(np.median(full)),'p95':float(np.percentile(full,95))} if full else None,batch_scope='Fixed physical capacity; tail duplicates first real row, dummy outputs discarded. Weighted throughput counts only real images. Batch latency is not single-image latency.')
        return result
    finally:
        if sampler:sampler.close()
        if model:model.close()
        for image in cache.values():image.close()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train',help='Experimental image/text-only compact training');train.add_argument('--config',type=Path,required=True)
    export=sub.add_parser('export',help='Export BOTH graphs from a typed compact checkpoint');export.add_argument('--checkpoint',type=Path,required=True);export.add_argument('--deployment',type=Path,required=True);export.add_argument('--static-batch',type=int,choices=[1,2,4,8],default=None,help='Separate fixed-capacity ASCII2048→512 export; omit for original B1/Unicode B1 path')
    build=sub.add_parser('build',help='Build dedicated compact engines on this GPU');build.add_argument('--deployment',type=Path,required=True)
    for command in ['infer','evaluate','benchmark']:
        p=sub.add_parser(command);p.add_argument('--backend',choices=['torch','onnx','tensorrt'],default='torch');p.add_argument('--checkpoint',type=Path);p.add_argument('--deployment',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--output',type=Path)
        if command=='infer':group=p.add_mutually_exclusive_group(required=True);group.add_argument('--image',type=Path);group.add_argument('--folder',type=Path)
        else:p.add_argument('--folder',type=Path,required=True)
        if command=='benchmark':p.add_argument('--repeats',type=int,default=3);p.add_argument('--predecoded',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='train':
            config=_config(args.config)
            from .waterec_compact_workflow import train_compact
            result=train_compact(config)
        elif args.command=='export':
            if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
            if args.static_batch is None:
                from .waterec_compact_export import export_compact
                deployment=export_compact(args.checkpoint,args.deployment)
            else:
                from .waterec_compact_batch_export import export_compact_batch
                deployment=export_compact_batch(args.checkpoint,args.deployment,args.static_batch)
            result={'deployment':str(deployment),'scope':'CPUexport only; targetdevice parity required'}
        elif args.command=='build':
            metadata=_metadata(args.deployment)
            if metadata['max_tokens']>2048:raise ValueError('Compact FP16 TensorRT budget must be64..2048; native/CPU ONNX budgets are separate')
            if metadata['family']=='waterec_compact_batch':
                from .waterec_compact_batch_build import build_compact_batch
                deployment=build_compact_batch(args.deployment)
            else:
                from .waterec_compact_build import build_compact
                deployment=build_compact(args.deployment)
            result={'deployment':str(deployment),'scope':'Build only; exactpixels/tokens/raw andtaskaccuracy qualification beforetiming'}
        else:result=_predict(args)
    except (ValueError,FileNotFoundError,NotADirectoryError,FileExistsError) as error:parser.error(str(error))
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if getattr(args,'output',None):args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text,end='');return 0
if __name__=='__main__':main()
