# SPDX-License-Identifier: GPL-3.0-or-later
"""Index existing official image-only preprocessing; no label arrays are read."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();records=[]
cases=json.loads((w/'inputs/cases.json').read_text());run=json.loads((w/'reference-point/run.json').read_text())
if not cases['complete'] or not run['complete']:raise ValueError('Incomplete inputs/reference run')
for c in cases['cases']:
 n=c['case'];m=json.loads((w/'reference-point/preprocessed'/f'{n}.json').read_text())
 if m['reference_mask_used_for_preprocessing'] or c['primary_prompt_uses_reference_mask']:raise ValueError('Reference-dependent preprocessing')
 records.append({'case':n,'patient_id':c['patient_id'],'scan_id':c['scan_id'],'image':f'preprocessed/{n}.npy','image_metadata':f'preprocessed/{n}.json','image_shape_CZYX':m['shape_CZYX'],'image_dtype':'float32','configuration':'3d_fullres','target_spacing_ZYX':m['target_spacing_ZYX']})
f=w/'reference-point/prepared-dataset.json'
if f.exists():raise ValueError('Fresh index required')
f.write_text(json.dumps({'complete':True,'case_count':len(records),'array_axis_order':'CZYX','image_representation':'float32 CTNormalization output on official LesionLocator resampling grid','source_code_commit':'ef073baaa0dcba8dd036ec5a29f130db18c9d580','cases':records},indent=2)+'\n')
