# SPDX-License-Identifier: GPL-3.0-or-later
# Release adaptation: explicit workspace/model arguments; numerical extraction unchanged.
#!/usr/bin/env python3
"""Copy frozen cached teacher masks to native feature-channel files; IO only."""
import argparse, json, time
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def geometry(image):
    return {'originLPS':list(image.GetOrigin()),'spacingXYZ':list(image.GetSpacing()),'directionLPS':list(image.GetDirection()),'sizeXYZ':list(image.GetSize())}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workspace',type=Path,required=True);args=parser.parse_args();BASE=args.workspace.resolve()
    source=json.loads((BASE/'inputs/cases.json').read_text());run=json.loads((BASE/'reference-point/run.json').read_text())
    if not source['complete'] or not source['complete_cohort'] or source['reference_center_prompts_enabled'] or not run['complete'] or run['reference_masks_used_as_model_inputs']:
        raise ValueError('Complete independent-point frozen teacher provenance required')
    verified=json.loads((BASE/'point-model-download-complete.json').read_text())
    if len(verified)!=5 or any(x['status']!='verified' for x in verified):raise ValueError('Five verified frozen checkpoint receipts required')
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'arrays').mkdir(exist_ok=True)
    manifest={'format':'segmentation-additional-channels-1','complete':False,'array_axis_order':'CZYX','frozen_network':True,'reference_labels_used_for_features':False,'independent_seed':True,'model_provenance':{'name':'LesionLocator point_optimized','upstream_commit':'ef073baaa0dcba8dd036ec5a29f130db18c9d580','source_repository':'https://github.com/MIC-DKFZ/LesionLocator','model_record':'https://zenodo.org/records/15174217','source_license':'GPL-3.0','weight_license':'CC-BY-NC-SA-4.0','folds':[0,1,2,3,4],'mirror_views':8,'frozen_pretrained_weights':True,'local_reference_labels_used_for_network_training_or_feature_extraction':False,'external_checkpoint_training_membership':'unavailable; RIDER overlap unverified','checkpoint_provenance_file':str(BASE/'point-model-download-complete.json'),'checkpoint_sha256':[x['member_sha256'] for x in verified],'source_reference_run':str(BASE/'reference-point/run.json')},'cases':[]}
    start=time.monotonic()
    for case in source['cases']:
        if case['primary_prompt_uses_reference_mask']:raise ValueError('Reference-derived prompt prohibited')
        name=case['case'];mask_path=BASE/'reference-point/predictions'/f'{name}.mha';out_path=args.out/'arrays'/f'{name}.npy'
        if out_path.exists():raise ValueError('Refuse overwriting feature cache: '+str(out_path))
        image=sitk.ReadImage(str(mask_path));array=sitk.GetArrayViewFromImage(image)
        if array.dtype!=np.uint8 or int(array.min())<0 or int(array.max())>1:raise ValueError('Expected unchanged uint8 binary teacher mask')
        reader=sitk.ImageFileReader();reader.SetFileName(str((BASE/'inputs'/case['image_path']).resolve()) if not Path(case['image_path']).is_absolute() else case['image_path']);reader.ReadImageInformation()
        if geometry(reader)!=geometry(image):raise ValueError('Teacher geometry differs from independent source image')
        values=array.astype(np.float32)[None]
        tmp=out_path.with_suffix('.tmp.npy');np.save(tmp,values,allow_pickle=False);tmp.replace(out_path)
        manifest['cases'].append({'scan_id':case['scan_id'],'patient_id':case['patient_id'],'case':name,'source_mask':str(mask_path),'source_mask_bytes':mask_path.stat().st_size,'mask_values_unmodified':True,'arrays':[{'array':str(out_path.resolve()),'shape_CZYX':list(values.shape),'channel_names':['cnn_lesionlocator_mask'],'channel_kinds':['binary_mask'],'interpolation':'nearest','outside_values':[0],'grid':geometry(image)}]})
        atomic_json(args.out/'channels.json',manifest)
        print(json.dumps({'event':'cached_mask_channel_saved','case':name,'completed':len(manifest['cases']),'requested':len(source['cases']),'seconds':round(time.monotonic()-start,1)}),flush=True)
        del values,array,image
    manifest['complete']=True;manifest['seconds']=time.monotonic()-start;manifest['model_inference_performed']=False;manifest['numerical_training_performed']=False
    atomic_json(args.out/'channels.json',manifest);print(json.dumps({'complete':True,'index':str(args.out/'channels.json'),'seconds':round(manifest['seconds'],1)}),flush=True)

if __name__=='__main__':main()
