# SPDX-License-Identifier: GPL-3.0-or-later
# Release adaptation: explicit workspace/model arguments; numerical extraction unchanged.
#!/usr/bin/env python3
"""Publish complete62-case additional-channel index using metadata/header checks only."""
import argparse,json
from pathlib import Path
import numpy as np

NAMES=['cnn_lesionlocator_foreground_probability']+[f'cnn_decoder_fold0_channel_{i:02d}' for i in range(32)]

def need(yes,message):
    if not yes:raise ValueError(message)

def header(array,dtype,shape):
    path=Path(array);need(path.exists(),'Missing feature file: '+str(path))
    values=np.load(path,mmap_mode='r',allow_pickle=False)
    need(values.dtype==np.dtype(dtype) and list(values.shape)==shape,'Feature shape/dtype differs: '+str(path))
    need(path.stat().st_size==values.offset+values.nbytes,'Feature NPY extent differs: '+str(path))
    del values

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workspace',type=Path,required=True);args=parser.parse_args();HERE=args.workspace.resolve()
    source=json.loads(args.source.read_text());masks=json.loads((HERE/'cached-mask/channels.json').read_text());cases=json.loads((HERE/'inputs/cases.json').read_text())['cases']
    need(source.get('complete') and source.get('whole_cohort_complete'),'Continuous extraction is not full-cohort complete; refusing final index')
    need(len(source.get('cases',[]))==len(cases)==62,'Complete 62-case coverage required')
    need(masks.get('complete') is True and len(masks.get('cases',[]))==62,'Complete cached mask index required')
    need(source.get('array_axis_order')=='CZYX','Explicit CZYX order required')
    for key in ('frozen_network','independent_seed'):
        need(source.get(key) is True,'Frozen independent network provenance required')
    need(source.get('reference_labels_used_for_features') is False,'No local reference feature inputs required')
    expected={int(c['scan_id']):c for c in cases};binary={int(c['scan_id']):c for c in masks['cases']}
    observed={int(c['scan_id']):c for c in source['cases']};need(len(observed)==62 and set(observed)==set(expected),'Duplicate or missing scan')
    audited=[]
    for original in cases:
        scan=int(original['scan_id']);case=observed[scan];need(case['patient_id']==original['patient_id'] and case['case']==original['case'],'Patient/case identity differs')
        arrays=case['arrays'];need(len(arrays)==2 and arrays[0]==binary[scan]['arrays'][0],'Cached teacher mask provenance changed')
        continuous=arrays[1];need(continuous['shape_CZYX']==[33,80,80,80] and continuous['channel_names']==NAMES and continuous['channel_kinds']==['probability']+['learned']*32,'Continuous schema/order differs')
        need(continuous['interpolation']=='linear' and continuous['outside_values']==[0]*33 and continuous['grid']==original['cropGrid'],'Continuous physical grid or outside convention differs')
        for array in arrays:header(array['array'],'float32',array['shape_CZYX'])
        support=case['continuous_feature_support'];header(support['mask_array'],'uint8',[1,80,80,80]);need(support['valid_voxels']+support['invalid_voxels']==512000 and support['invalid_zeros_are_not_physically_valid_activations'],'Explicit unknown support required')
        receipt=json.loads((args.source.parent/'case-receipts'/f"{case['case']}.json").read_text())
        need(receipt['complete'] and receipt['frozen_probability_folds']==[0,1,2,3,4] and receipt['mirror_views']==8 and receipt['learned_fold']==0 and receipt['learned_view']=='unmirrored' and receipt['reference_labels_used_for_features'] is False,'Incomplete or unfrozen case receipt')
        need(receipt['manifest_case']==case,'Case receipt/index differs')
        need(receipt['shape_CZYX']==[33,80,80,80] and receipt['feature_names']==NAMES and receipt['grid']==original['cropGrid'],'Case receipt output schema differs')
        need(receipt['network_forward_calls']==40,'Frozen five-fold/eight-view forward count differs')
        audited.append({'scan_id':scan,'patient_id':case['patient_id'],'inference_seconds':receipt['inference_seconds'],'postprocessing_seconds':receipt['postprocessing_seconds'],'network_forward_calls':receipt['network_forward_calls'],'valid_voxels':support['valid_voxels'],'invalid_voxels':support['invalid_voxels'],'peak_torch_GPU_allocated_bytes':receipt['peak_torch_GPU_allocated_bytes'],'peak_torch_GPU_reserved_bytes':receipt['peak_torch_GPU_reserved_bytes']})
    final=dict(source);final['cases']=[observed[int(c['scan_id'])] for c in cases];final['complete']=True;final['validated_original_feature_count']=20;final['additional_channel_count']=34;final['expected_combined_feature_count']=54;final['full_cohort_metadata_validation']={'complete':True,'cases':62,'patient_count':len({c['patient_id'] for c in cases}),'feature_payloads_reread':False,'reference_label_payloads_read':False,'GPU_calls_performed':False,'headers_and_extents_verified':True,'source_case_receipts_verified':True}
    tmp=args.out.with_suffix(args.out.suffix+'.tmp');tmp.write_text(json.dumps(final,indent=2)+'\n');tmp.replace(args.out)
    audit={'format':'rider-frozen-cnn-extraction-audit-1','complete':True,'cases':audited,'inference_seconds_sum':sum(c['inference_seconds'] for c in audited),'postprocessing_seconds_sum':sum(c['postprocessing_seconds'] for c in audited),'peak_torch_GPU_allocated_bytes':max(c['peak_torch_GPU_allocated_bytes'] for c in audited),'peak_torch_GPU_reserved_bytes':max(c['peak_torch_GPU_reserved_bytes'] for c in audited),'all_nominal40_network_forwards':all(c['network_forward_calls']==40 for c in audited),'valid_ROI_voxels':sum(c['valid_voxels'] for c in audited),'invalid_ROI_voxels':sum(c['invalid_voxels'] for c in audited),'full_manifest':str(args.out),'accuracy_or_independent_generalization_claim':False}
    args.out.with_name('full-extraction-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'complete':True,'cases':62,'combined_features':54,'index':str(args.out),'inference_seconds_sum':round(audit['inference_seconds_sum'],1),'all40_forwards':audit['all_nominal40_network_forwards']},indent=2))

if __name__=='__main__':main()
