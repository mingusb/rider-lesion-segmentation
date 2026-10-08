# SPDX-License-Identifier: GPL-3.0-or-later
# Release adaptation: explicit workspace/model arguments; numerical extraction unchanged.
#!/usr/bin/env python3
"""Frozen official fivefold/TTA inference plus fixedfold0 learned ROI channels.
Only local experiment IO and sanctioned official PyTorch pretrained inference.
No reference labels, fitting, extra network forwards, or score-driven crops.
"""
import argparse, gc, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from lesionlocator.inference.lesionlocator_segment import LesionLocatorSegmenter
from lesionlocator.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from lesionlocator.utilities.prompt_handling.prompt_handler import sparse_to_dense_prompt

FOLDS=[0,1,2,3,4]
NAMES=['cnn_lesionlocator_foreground_probability']+[f'cnn_decoder_fold0_channel_{i:02d}' for i in range(32)]
KINDS=['probability']+['learned']*32

def atomic_json(path,record):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)

def physical_grid(metadata, transpose_forward, cropped_start, cropped_shape):
    """Map saved resized voxel centers to original image physical LPS geometry."""
    props=metadata['properties'];sitk=props['sitk_stuff'];direction=np.asarray(sitk['direction'],dtype=np.float64).reshape(3,3)
    original_spacing=np.asarray(sitk['spacing'],dtype=np.float64)
    old_shape=np.asarray(props['shape_after_cropping_and_before_resampling'],dtype=np.float64)
    new_shape=np.asarray(metadata['shape_CZYX'][1:],dtype=np.float64)
    ratio=old_shape/new_shape
    crop_start=np.asarray([x[0] for x in props['bbox_used_for_cropping']],dtype=np.float64)
    prep_offset=crop_start+(ratio-1)*.5+np.asarray(cropped_start)*ratio
    original_offset=np.zeros(3,dtype=np.float64);new_direction=np.zeros((3,3),dtype=np.float64);new_spacing=np.zeros(3,dtype=np.float64)
    for prep_axis,original_zyx_axis in enumerate(transpose_forward):
        original_xyz_axis=2-original_zyx_axis;new_xyz_axis=2-prep_axis
        original_offset[original_xyz_axis]=prep_offset[prep_axis]
        new_spacing[new_xyz_axis]=original_spacing[original_xyz_axis]*ratio[prep_axis]
        new_direction[:,new_xyz_axis]=direction[:,original_xyz_axis]
    origin=np.asarray(sitk['origin'],dtype=np.float64)+direction@(original_offset*original_spacing)
    return {'originLPS':origin.tolist(),'spacingXYZ':new_spacing.tolist(),'directionLPS':new_direction.reshape(-1).tolist(),'sizeXYZ':list(reversed(cropped_shape))}


def roi_sample_grid(metadata, transpose, score_grid, patch):
    """Return precise FP32 patch voxel-center sampling grid and explicit support."""
    full_shape=metadata['shape_CZYX'][1:]
    prepped=physical_grid(metadata,transpose,[0,0,0],full_shape)
    shape=list(reversed(score_grid['sizeXYZ']))
    zz,yy,xx=np.meshgrid(np.arange(shape[0],dtype=np.float64),np.arange(shape[1],dtype=np.float64),np.arange(shape[2],dtype=np.float64),indexing='ij')
    xyz=np.stack((xx,yy,zz),axis=-1)
    score_direction=np.asarray(score_grid['directionLPS'],dtype=np.float64).reshape(3,3)
    world=np.asarray(score_grid['originLPS'])+(xyz*np.asarray(score_grid['spacingXYZ']))@score_direction.T
    prepped_direction=np.asarray(prepped['directionLPS']).reshape(3,3)
    prepped_xyz=((world-np.asarray(prepped['originLPS']))@np.linalg.inv(prepped_direction).T)/np.asarray(prepped['spacingXYZ'])
    patch_start=np.asarray(patch['patch_start'][::-1],dtype=np.float64);padding=np.asarray(patch['padding_before'][::-1],dtype=np.float64)
    patch_size=np.asarray(patch['patch_stop'][::-1],dtype=np.float64)-patch_start
    patch_xyz=prepped_xyz+padding-patch_start
    # A value beyond either voxel-center support is unknown, not a valid zero activation.
    valid=np.all((prepped_xyz>=0)&(prepped_xyz<=np.asarray(full_shape[::-1])-1),axis=-1)&np.all((patch_xyz>=0)&(patch_xyz<=patch_size-1),axis=-1)
    normalized=(2*patch_xyz/(patch_size-1)-1).astype(np.float32)
    return torch.from_numpy(normalized[None]),torch.from_numpy(valid[None,None]),prepped


class FrozenFeatureSegmenter(LesionLocatorSegmenter):
    """Read only first unmirroredfold0 activation; official40 predictions untouched."""
    def install_hook(self):
        self._hook_handle=self.network.decoder.stages[-1].register_forward_hook(self._decoder_hook)
        self.reset_case()

    def reset_case(self,metadata=None,score_grid=None):
        self._case_metadata=metadata;self._score_grid=score_grid;self._fold_count=0;self._network_forwards=0
        self._patch_geometry=None;self._sample_grid=None;self._valid_support=None;self._learned_roi=None;self._tta_calls=0

    @torch.inference_mode()
    def _decoder_hook(self,module,inputs,output):
        if output.ndim!=5 or output.shape[0]!=1 or output.shape[1]!=32:raise RuntimeError('Unexpected final32-channel decoder activation')
        if tuple(output.shape[2:])!=tuple(self.configuration_manager.patch_size):raise RuntimeError('Decoder spatial shape differs from frozen neural patch')
        if self._fold_count==0 and self._tta_calls==0 and self._learned_roi is None:
            grid=self._sample_grid.to(output.device);valid=self._valid_support.to(output.device);chunks=[]
            # FP32 spatial coordinates and arithmetic; bound cast scratch to4channels.
            with torch.autocast(output.device.type,enabled=False):
                for offset in range(0,32,4):
                    sampled=F.grid_sample(output[:,offset:offset+4].float(),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
                    sampled.masked_fill_(~valid,0);chunks.append(sampled[0].cpu())
                    del sampled
            self._learned_roi=torch.cat(chunks,dim=0);del chunks,grid,valid
        self._tta_calls+=1;self._network_forwards+=1

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self,x):
        self._tta_calls=0
        output=super()._internal_maybe_mirror_and_predict(x)
        expected=2**len(self.allowed_mirroring_axes) if self.use_mirroring else 1
        if self._tta_calls!=expected:raise RuntimeError('Incomplete official mirroring prediction coverage')
        return output

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self,input_image,dense_prompt):
        old_shape=np.asarray(input_image.shape[1:],dtype=np.int64)
        padded_shape=np.maximum(old_shape,np.asarray(self.configuration_manager.patch_size,dtype=np.int64))
        padding_before=(padded_shape-old_shape)//2
        slicers=self._internal_get_sliding_window_slicers(tuple(map(int,padded_shape)),dense_prompt)
        if len(slicers)!=1:raise RuntimeError('Independent point extractor requires exactly one official neural patch')
        geometry={'padding_before':padding_before.tolist(),'original_shape':old_shape.tolist(),'patch_start':[int(x.start) for x in slicers[0][1:]],'patch_stop':[int(x.stop) for x in slicers[0][1:]]}
        if self._patch_geometry is not None and geometry!=self._patch_geometry:raise RuntimeError('Patch geometry differs across frozen folds')
        if self._sample_grid is None:
            self._sample_grid,self._valid_support,self._prepped_grid=roi_sample_grid(self._case_metadata,self.plans_manager.transpose_forward,self._score_grid,geometry)
        self._patch_geometry=geometry
        result=super().predict_sliding_window_return_logits(input_image,dense_prompt)
        self._fold_count+=1
        return result

    @torch.inference_mode()
    def compact_outputs(self,logits):
        if self._fold_count!=5 or self._learned_roi is None:raise RuntimeError('Expected5fold probability plus fixedfold0 latent channels')
        g=self._patch_geometry
        # Remove all uncomputed NaNs outside the one official patch before sampling.
        starts=[max(0,a-p) for a,p in zip(g['patch_start'],g['padding_before'])]
        stops=[min(n,a-p) for n,a,p in zip(g['original_shape'],g['patch_stop'],g['padding_before'])]
        inside=logits[(slice(None),*[slice(a,b) for a,b in zip(starts,stops)])]
        if not torch.isfinite(inside).all():raise RuntimeError('Nonfinite logits inside inferred patch')
        # Place valid native logits in a small neural-patch tensor, never full-volume probability arrays.
        patch_shape=[b-a for a,b in zip(g['patch_start'],g['patch_stop'])]
        patch_logits=torch.zeros((2,*patch_shape),dtype=torch.float32,device=self.device)
        patch_slices=[slice(a+p-origin,b+p-origin) for a,b,p,origin in zip(starts,stops,g['padding_before'],g['patch_start'])]
        patch_logits[(slice(None),*patch_slices)]=inside.to(device=self.device,dtype=torch.float32)
        roi_grid=self._sample_grid.to(self.device);valid=self._valid_support.to(self.device)
        with torch.autocast(self.device.type,enabled=False):
            sampled_logits=F.grid_sample(patch_logits[None],roi_grid,mode='bilinear',padding_mode='zeros',align_corners=True)[0]
        probability=self.label_manager.apply_inference_nonlin(sampled_logits)[1:2].float()
        probability.masked_fill_(~valid[0],0)
        channels=torch.cat((probability.cpu(),self._learned_roi.float()),dim=0).contiguous()
        if not torch.isfinite(channels).all():raise RuntimeError('Nonfinite probability/learned ROI features')
        if channels[0].min()<0 or channels[0].max()>1:raise RuntimeError('Invalid frozen foreground probability')
        return channels.numpy(),self._valid_support[0].numpy().astype(np.uint8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--check-cached-mask',action='store_true',help='Compare first fresh official mask to cached teacher prediction, never target labels')
    parser.add_argument('--metadata-only',action='store_true',help='Validate cached input/score geometry without a GPU context')
    args=parser.parse_args();BASE=args.workspace.resolve()
    source=json.loads((BASE/'inputs/cases.json').read_text());cached_masks=json.loads((BASE/'cached-mask/channels.json').read_text())
    if not source['complete'] or source['reference_center_prompts_enabled'] or not cached_masks['complete']:raise ValueError('Complete independent point inputs and frozen masks required')
    cases=source['cases'][args.start:]
    if args.limit:cases=cases[:args.limit]
    model=args.model.resolve();plans=json.loads((model/'plans.json').read_text())
    if args.metadata_only:
        for case in cases:
            meta=json.loads((BASE/'reference-point/preprocessed'/f"{case['case']}.json").read_text())
            if case['primary_prompt_uses_reference_mask'] or meta['reference_mask_used_for_preprocessing'] or len(meta['transformed_points_ZYX'])!=1:raise ValueError('Independent cached point required')
            score=case['cropGrid']
            if score['sizeXYZ']!=[80,80,80] or score['spacingXYZ']!=[2,2,2]:raise ValueError('Fixed80cubed/2mm score grid required')
            grid=physical_grid(meta,plans['transpose_forward'],[0,0,0],meta['shape_CZYX'][1:])
            if not all(x>0 and math.isfinite(x) for x in grid['spacingXYZ']):raise ValueError('Invalid resize geometry')
        print(json.dumps({'metadata_validated_cases':len(cases),'GPU_inference_performed':False,'probability_channels':1,'learned_channels':32,'learned_source':'unmirrored firstfold0 final decoder','output_shape_CZYX':[33,80,80,80],'frozen_probability_folds':FOLDS,'mirror_views':8},indent=2));return
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'arrays').mkdir(exist_ok=True);(args.out/'case-receipts').mkdir(exist_ok=True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(args.threads)
    predictor=FrozenFeatureSegmenter(use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda'),allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model),FOLDS);predictor.install_hook()
    provenance=dict(cached_masks['model_provenance']);provenance.update({'probability':'official5fold/8mirror averaged logits, FP32 linear sampling on fixed physical score grid, then official inference softmax','learned_channels':'all32 final decoder channels from fixedfold0 first unmirrored forward; no feature/projection fitting or additional neural forward','neural_patch':'unchanged official independent point restriction','feature_support':'outside saved preprocessed-array or inferred neural-patch voxel-center bounds is invalid; zero-filled and separately flagged','reference_labels_used_for_features':False,'output_grid':'same independently declared80cubed,2mm physical score grid as existing20 features','required_pretrained_network_at_inference':True})
    manifest={k:v for k,v in cached_masks.items() if k not in ('cases','complete','seconds','model_inference_performed')};manifest.update({'complete':False,'requested_cases':len(cases),'model_provenance':provenance,'cases':[]})
    masks={int(c['scan_id']):c for c in cached_masks['cases']};start=time.monotonic();fresh=0
    for case in cases:
        name=case['case'];receipt_path=args.out/'case-receipts'/f'{name}.json';array_path=args.out/'arrays'/f'{name}.npy';support_path=args.out/'arrays'/f'{name}.support.npy'
        if receipt_path.exists():
            saved=json.loads(receipt_path.read_text())
            if saved.get('complete') and array_path.exists() and support_path.exists() and saved.get('feature_names')==NAMES:
                manifest['cases'].append(saved['manifest_case']);atomic_json(args.out/'channels.json',manifest);continue
            raise ValueError('Existing incomplete or different-schema receipt; preserve/use fresh output')
        if array_path.exists() or support_path.exists():raise ValueError('Refuse overwriting unreceipted features')
        meta=json.loads((BASE/'reference-point/preprocessed'/f'{name}.json').read_text())
        if case['primary_prompt_uses_reference_mask'] or meta['reference_mask_used_for_preprocessing']:raise ValueError('Reference-dependent input prohibited')
        array=np.load(BASE/'reference-point/preprocessed'/f'{name}.npy',allow_pickle=False)
        if list(array.shape)!=meta['shape_CZYX'] or array.dtype!=np.float32:raise ValueError('Cached CT shape/dtype differs')
        tensor=torch.from_numpy(array);points=meta['transformed_points_ZYX']
        if len(points)!=1:raise ValueError('One independent point required')
        dense=sparse_to_dense_prompt(points[0],'point',array=tensor);predictor.reset_case(meta,case['cropGrid']);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        print(json.dumps({'event':'frozen_CNN_feature_inference_started','case':name,'folds':FOLDS,'mirror_views':8,'learned_source':'fold0 first unmirrored forward'}),flush=True)
        logits=predictor.predict_logits_from_preprocessed_data(tensor,dense).cpu();torch.cuda.synchronize();inference_seconds=time.monotonic()-t
        channels,support=predictor.compact_outputs(logits);mask_agreement=None
        if args.check_cached_mask and fresh==0:
            import SimpleITK as sitk
            mask=convert_predicted_logits_to_segmentation_with_correct_shape(logits,predictor.plans_manager,predictor.configuration_manager,predictor.label_manager,meta['properties'],num_threads_torch=8)
            if isinstance(mask,torch.Tensor):mask=mask.cpu().numpy()
            old=sitk.GetArrayFromImage(sitk.ReadImage(str(BASE/'reference-point/predictions'/f'{name}.mha')))
            if mask.shape!=old.shape:raise RuntimeError('Cached teacher export shape mismatch')
            mask_agreement={'compared_to':'cached teacher prediction, not target reference','different_voxels':int(np.count_nonzero(mask!=old)),'voxels':int(old.size)};del mask,old
        for destination,values in ((array_path,channels),(support_path,support)):
            tmp=destination.with_suffix('.tmp.npy');np.save(tmp,values,allow_pickle=False);tmp.replace(destination)
        channel_array={'array':str(array_path.resolve()),'shape_CZYX':list(channels.shape),'channel_names':NAMES,'channel_kinds':KINDS,'interpolation':'linear','outside_values':[0]*33,'grid':case['cropGrid']}
        valid_count=int(np.count_nonzero(support));support_record={'mask_array':str(support_path.resolve()),'shape_CZYX':list(support.shape),'valid_voxels':valid_count,'invalid_voxels':int(support.size)-valid_count,'invalid_values':[0]*33,'invalid_zeros_are_not_physically_valid_activations':True}
        manifest_case={'scan_id':case['scan_id'],'patient_id':case['patient_id'],'case':name,'arrays':masks[int(case['scan_id'])]['arrays']+[channel_array],'continuous_feature_support':support_record}
        receipt={'complete':True,'case':name,'scan_id':case['scan_id'],'patient_id':case['patient_id'],'inference_seconds':inference_seconds,'postprocessing_seconds':time.monotonic()-t-inference_seconds,'peak_torch_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_torch_GPU_reserved_bytes':torch.cuda.max_memory_reserved(),'neural_patch':predictor._patch_geometry,'shape_CZYX':list(channels.shape),'grid':case['cropGrid'],'feature_names':NAMES,'frozen_probability_folds':FOLDS,'mirror_views':8,'learned_fold':0,'learned_view':'unmirrored','network_forward_calls':predictor._network_forwards,'reference_labels_used_for_features':False,'cached_teacher_agreement':mask_agreement,'continuous_feature_support':support_record,'manifest_case':manifest_case}
        atomic_json(receipt_path,receipt);manifest['cases'].append(manifest_case);atomic_json(args.out/'channels.json',manifest);fresh+=1
        print(json.dumps({'event':'frozen_CNN_feature_case_complete','case':name,'seconds':round(inference_seconds,1),'completed':len(manifest['cases']),'requested':len(cases),'GPU_peak_GiB':round(receipt['peak_torch_GPU_allocated_bytes']/2**30,2),'network_forwards':predictor._network_forwards,'valid_ROI_voxels':valid_count,'cached_teacher_agreement':mask_agreement}),flush=True)
        del logits,channels,support,dense,tensor,array;predictor.reset_case();torch.cuda.empty_cache();gc.collect()
    manifest['complete']=True;manifest['seconds']=time.monotonic()-start;manifest['model_inference_performed']=True;manifest['numerical_training_performed']=False;manifest['whole_cohort_complete']=len(manifest['cases'])==len(source['cases'])
    atomic_json(args.out/'channels.json',manifest);print(json.dumps({'complete':True,'whole_cohort_complete':manifest['whole_cohort_complete'],'index':str(args.out/'channels.json'),'seconds':round(manifest['seconds'],1)}),flush=True)

if __name__=='__main__':main()
