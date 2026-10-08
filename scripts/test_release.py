#!/usr/bin/env python3
"""Host-only packaging contracts; does not claim clinical or CUDA reproduction."""
import ast,hashlib,json,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Release(unittest.TestCase):
 def test_source_snapshot(self):
  pin=json.loads((ROOT/'protocol/core-source.json').read_text())
  for f,h in pin['source_files'].items():
   self.assertEqual(hashlib.sha256((ROOT/'vendor/decision-programs'/f).read_bytes()).hexdigest(),h,f)
 def test_plans(self):
  cv=json.loads((ROOT/'plans/latest54-cv.json').read_text());final=json.loads((ROOT/'plans/latest54-final.json').read_text());paired=json.loads((ROOT/'plans/cv-b8-dense161.json').read_text())
  self.assertEqual(len(cv['base_hyperparameters']),8);self.assertEqual(len(cv['meta_hyperparameters']),4)
  self.assertEqual([cv[k] for k in ['outer_folds','inner_folds','oof_folds']],[5,5,3]);self.assertEqual(cv['fit_group_weights'],'equal_patient');self.assertEqual(cv['fit_sampling_weights'],'inverse_retention')
  cuts=cv['native_prediction_thresholds'];self.assertEqual(len(cuts),161);self.assertEqual(len({struct.pack('f',x) for x in cuts}),161)
  for q in [final,paired]:
   for k in ['base_hyperparameters','meta_hyperparameters','native_prediction_thresholds','reference_predictor','fit_group_weights','fit_sampling_weights']:self.assertEqual(q[k],cv[k])
  self.assertNotIn('final_deployment',cv);self.assertEqual(final['final_deployment'],{'convert':False});self.assertEqual(paired['rule_search_views'][0]['native_prediction_candidate_indices'],list(range(20)))
 def test_cohort(self):
  d=json.loads((ROOT/'protocol/cohort.json').read_text());self.assertEqual(len(d['scans']),62);self.assertEqual(len({s['patient_id'] for s in d['scans']}),32);self.assertEqual(len({s['scan_id'] for s in d['scans']}),62)
  self.assertEqual(d['crop_width_mm']/d['grid_spacing_mm'],80);self.assertEqual(d['training_sample_stride'],32)
 def test_syntax(self):
  for p in (ROOT/'scripts').rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 def test_runtime_build_target(self):
  tree=ast.parse((ROOT/'scripts/reproduce.py').read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run']
  build=[n for n in calls if any(isinstance(x,ast.Constant) and x.value=='--build' for x in n.args)]
  self.assertEqual(len(build),1)
  self.assertIn('decision_programs_model_tools',[x.value for x in build[0].args if isinstance(x,ast.Constant)])
 def test_stable_export_source(self):
  tree=ast.parse((ROOT/'scripts/reproduce.py').read_text())
  exports=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='run' and any(isinstance(x,ast.Constant) and x.value=='segmentation-export' for x in node.args):self.fail('Temporary frontend export descriptor is not persistent')
   if isinstance(node,ast.Call) and 'class_segmentation_export' in ast.unparse(node):exports.append(node)
  self.assertTrue(exports)
 def test_prepared_index_schema(self):
  with tempfile.TemporaryDirectory() as d:
   w=Path(d);(w/'inputs').mkdir();(w/'reference-point/preprocessed').mkdir(parents=True)
   case={'case':'case0','scan_id':9,'patient_id':4,'primary_prompt_uses_reference_mask':False}
   (w/'inputs/cases.json').write_text(json.dumps({'complete':True,'cases':[case]}));(w/'reference-point/run.json').write_text('{"complete":true}')
   metadata={'case':'case0','reference_mask_used_for_preprocessing':False,'shape_CZYX':[1,4,5,6],'target_spacing_ZYX':[1,.8,.8]}
   (w/'reference-point/preprocessed/case0.json').write_text(json.dumps(metadata))
   subprocess.run([sys.executable,str(ROOT/'scripts/lesionlocator/make_prepared_index.py'),'--workspace',str(w)],check=True)
   index=json.loads((w/'reference-point/prepared-dataset.json').read_text())
   self.assertEqual(index['image_representation'],'float32 CTNormalization output on official LesionLocator resampling grid')
   self.assertEqual(index['array_axis_order'],'CZYX');self.assertEqual(index['cases'][0]['configuration'],'3d_fullres');self.assertEqual(index['cases'][0]['image_shape_CZYX'],metadata['shape_CZYX'])
   self.assertNotIn('reference_labels',index['cases'][0])
if __name__=='__main__':unittest.main()
