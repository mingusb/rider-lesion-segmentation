#!/usr/bin/env python3
"""Export complete primary nested metrics without selecting by held-out outcomes."""
import argparse,csv,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=json.loads(a.result.read_text())
if not r.get('complete') or not r.get('native_evaluation_complete'):raise ValueError('Completed native evaluation required')
names=['XGBoost','Nonlinear combiner','Inner-selected family','Fixed CNN mask','Inner-selected CNN cutoff']
if r.get('metrics_order')!=['native_baseline','nonlinear_combiner','inner_selected','reference_default','reference_inner_selected'] or len(r['metrics'])!=5:raise ValueError('Expected released ordered five-method protocol')
records=[]; patient_set=None; scan_set=None
for name,m in zip(names,r['metrics']):
 if len(m['patients'])!=32 or len(m['scans'])!=62:raise ValueError('Full32patient/62scan coverage required')
 if len({x['patient_id'] for x in m['patients']})!=32 or len({x['scan_id'] for x in m['scans']})!=62:raise ValueError('Duplicate patient/scan')
 pids={x['patient_id'] for x in m['patients']};sids={(x['scan_id'],x['patient_id']) for x in m['scans']}
 if patient_set is not None and (pids!=patient_set or sids!=scan_set):raise ValueError('Method coverage differs')
 patient_set,scan_set=pids,sids;pooled={pid:[0,0,0,0] for pid in pids}
 for x in m['scans']:
  vals=[x['TP'],x['FP'],x['FN_in_ROI']+x['FN_outside_ROI'],x['TN_in_ROI']]
  if any(type(v) is not int or v<0 for v in vals):raise ValueError('Invalid scan count')
  for i,v in enumerate(vals):pooled[x['patient_id']][i]+=v
 for x in m['patients']:
  if pooled[x['patient_id']]!=[x[k] for k in ['TP','FP','FN','TN_in_ROI']]:raise ValueError('Scan/patient count mismatch')
  denom=2*x['TP']+x['FP']+x['FN'];dice=2*x['TP']/denom if denom else 1.
  if not math.isclose(dice,x['Dice'],rel_tol=0,abs_tol=1e-12):raise ValueError('Patient Dice/count mismatch')
  records.append(dict(method=name,**{k:x[k] for k in ['patient_id','TP','FP','FN','TN_in_ROI','Dice']}))
 if not math.isclose(sum(x['Dice'] for x in m['patients'])/32,m['macro_patient_Dice'],rel_tol=0,abs_tol=1e-12):raise ValueError('Macro/count mismatch')
if a.out.exists():raise ValueError('Fresh summary directory required')
a.out.mkdir(parents=True)
with (a.out/'patients.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary={'complete':True,'patients':32,'scans':62,'completed_native_fits':r['completed_native_fits'],'scores':{n:m['macro_patient_Dice'] for n,m in zip(names,r['metrics'])},'scope':'nested evaluation procedure; final all-patient refit has no new held-out score'}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
