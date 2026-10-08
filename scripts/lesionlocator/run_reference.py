# SPDX-License-Identifier: GPL-3.0-or-later
"""Experiment-only adapter calling unmodified, separately installed LesionLocator.
No training, candidate selection, or Decision Programs numerical code lives here.
"""
import argparse, gc, json, os, time
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from lesionlocator.inference.lesionlocator_segment import LesionLocatorSegmenter
from lesionlocator.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from lesionlocator.utilities.prompt_handling.prompt_handler import get_prompt_from_json, sparse_to_dense_prompt


def compatible(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):compatible(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [compatible(v) for v in value]
    return value


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(compatible(value),indent=2)+'\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--preprocess-only',action='store_true')
    p.add_argument('--state-name',default='run.json')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    for name in ['preprocessed','predictions','case-receipts']:(a.out/name).mkdir(exist_ok=True)
    record=json.loads(a.cases.read_text())
    cases=record['cases'][a.start:]
    if a.limit:cases=cases[:a.limit]
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(a.threads)
    predictor=LesionLocatorSegmenter(use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda'),allow_tqdm=False)
    if a.preprocess_only:
        from lesionlocator.utilities.plans_handling.plans_handler import PlansManager
        predictor.plans_manager=PlansManager(json.loads((a.model/'plans.json').read_text()))
        predictor.configuration_manager=predictor.plans_manager.get_configuration('3d_fullres')
        predictor.dataset_json=json.loads((a.model/'dataset.json').read_text())
    else:
        predictor.initialize_from_trained_model_folder(str(a.model),[0,1,2,3,4])
    pp=predictor.configuration_manager.preprocessor_class(verbose=False)
    state={'source':'Official LesionLocator ef073baaa0dcba8dd036ec5a29f130db18c9d580',
           'model':str(a.model),'folds':[0,1,2,3,4],'mirroring':True,'prompt':'independent RIDER spreadsheet point',
           'reference_masks_used_as_model_inputs':False,'requested_cases':len(cases),'cases':[], 'complete':False}
    write_json(a.out/a.state_name,state)
    for case in cases:
        case_id=case['case']
        receipt_path=a.out/'case-receipts'/f'{case_id}.json'
        prediction_path=a.out/'predictions'/f'{case_id}.mha'
        if receipt_path.exists():
            saved=json.loads(receipt_path.read_text())
            if saved.get('complete') and (a.preprocess_only or (saved.get('prediction_complete') and prediction_path.exists())):
                state['cases'].append(saved);write_json(a.out/a.state_name,state);continue
        t=time.monotonic()
        image_path=Path(case['image_path'])
        prompt_path=Path(case['prompt_path'])
        if not image_path.is_absolute():image_path=a.cases.parent/image_path
        if not prompt_path.is_absolute():prompt_path=a.cases.parent/prompt_path
        print(json.dumps({'stage':'preprocess','case':case_id}),flush=True)
        cache_array=a.out/'preprocessed'/f'{case_id}.npy'
        cache_metadata=a.out/'preprocessed'/f'{case_id}.json'
        reused_cache = not a.preprocess_only and cache_array.exists() and cache_metadata.exists()
        if reused_cache:
            cached=json.loads(cache_metadata.read_text())
            if cached['source_image']!=str(image_path) or cached['source_prompt']!=str(prompt_path):
                raise RuntimeError('Prepared cache source identity mismatch')
            if cached['target_spacing_ZYX']!=list(predictor.configuration_manager.spacing):
                raise RuntimeError('Prepared cache spacing mismatch')
            data=np.load(cache_array,allow_pickle=False)
            if list(data.shape)!=cached['shape_CZYX']:raise RuntimeError('Prepared cache shape mismatch')
            properties=cached['properties'];points=cached['transformed_points_ZYX']
        else:
            data,_,properties=pp.run_case([str(image_path)],None,predictor.plans_manager,predictor.configuration_manager,predictor.dataset_json)
            points=get_prompt_from_json(str(prompt_path),'point',properties,data.shape[1:])
        if len(points)!=1 or len(points[0])!=3:raise RuntimeError(f'{case_id}: expected one independent point')
        if not reused_cache:
            cache_tmp=cache_array.with_suffix('.tmp.npy')
            np.save(cache_tmp,data,allow_pickle=False)
            cache_tmp.replace(cache_array)
        preprocessing={'case':case_id,'shape_CZYX':list(data.shape),'properties':properties,
                       'target_spacing_ZYX':predictor.configuration_manager.spacing,'transformed_points_ZYX':points,
                       'source_image':str(image_path),'source_prompt':str(prompt_path),
                       'reference_mask_used_for_preprocessing':False}
        write_json(a.out/'preprocessed'/f'{case_id}.json',preprocessing)
        prep_seconds=time.monotonic()-t
        receipt={'case':case_id,'patient_id':case['patient_id'],'scan_id':case['scan_id'],'preprocessing_seconds':prep_seconds,
                 'preprocessed_data_saved':True,'prediction_complete':False,'complete':False}
        if not a.preprocess_only:
            print(json.dumps({'stage':'infer','case':case_id,'preprocessing_seconds':round(prep_seconds,1)}),flush=True)
            tensor=torch.from_numpy(data).to(dtype=torch.float32)
            dense=sparse_to_dense_prompt(points[0],'point',array=tensor)
            infer_t=time.monotonic()
            logits=predictor.predict_logits_from_preprocessed_data(tensor,dense).cpu()
            receipt['inference_seconds']=time.monotonic()-infer_t
            export_t=time.monotonic()
            mask=convert_predicted_logits_to_segmentation_with_correct_shape(logits,predictor.plans_manager,
                 predictor.configuration_manager,predictor.label_manager,properties,num_threads_torch=8)
            if isinstance(mask,torch.Tensor):mask=mask.cpu().numpy()
            if not np.isin(mask,[0,1]).all():raise RuntimeError(f'{case_id}: non-binary output')
            image=sitk.ReadImage(str(image_path))
            output=sitk.GetImageFromArray(mask.astype(np.uint8,copy=False));output.CopyInformation(image)
            tmp=prediction_path.with_name(prediction_path.stem+'.tmp.mha')
            sitk.WriteImage(output,str(tmp),False);tmp.replace(prediction_path)
            receipt.update(prediction_complete=True,prediction_positive_voxels=int(np.count_nonzero(mask)),export_seconds=time.monotonic()-export_t)
            del logits,mask,dense,tensor,output,image
            torch.cuda.empty_cache()
        receipt.update(complete=True,total_seconds=time.monotonic()-t)
        write_json(receipt_path,receipt);state['cases'].append(receipt);write_json(a.out/a.state_name,state)
        print(json.dumps({'stage':'complete','case':case_id,'seconds':round(receipt['total_seconds'],1),'finished':len(state['cases']),'requested':len(cases)}),flush=True)
        del data;gc.collect()
    state['complete']=True;write_json(a.out/a.state_name,state)

if __name__=='__main__':main()