#!/usr/bin/env python3
"""Orchestrate the released existing C++23/CUDA and external CNN tools."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def run(*args):
    print("+", " ".join(map(str,args)), flush=True)
    subprocess.run(list(map(str,args)),check=True)
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise ValueError(f"Refuse replacing {path}")
    path.write_text(json.dumps(obj,indent=2)+"\n")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["build","bind-cohort","prepare","study","convert"])
    p.add_argument("--work",type=Path,default=ROOT/"work")
    p.add_argument("--dicom-root",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--data",type=Path)
    p.add_argument("--library",type=Path)
    p.add_argument("--plan",type=Path,default=ROOT/"plans/latest54-cv.json")
    p.add_argument("--pipeline",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--check",action="store_true")
    p.add_argument("--architecture",default="86")
    a=p.parse_args(); w=a.work.resolve(); binary=w/"core/bin/decision-programs"
    if a.action=="build":
        run("cmake","-S",ROOT/"vendor/decision-programs","-B",w/"core","-G","Ninja","-DCMAKE_BUILD_TYPE=Release","-DDP_BUILD_DICOM=ON","-DDP_BUILD_REGIONAL_XAI=OFF","-DDP_INSTALL_PROOFS=OFF","-DBUILD_TESTING=ON",f"-DCMAKE_CUDA_ARCHITECTURES={a.architecture}")
        run("cmake","--build",w/"core","--target","decision-programs","class_study","class_segmentation_prepare","class_segmentation_export","class_segmentation_score","class_model_convert","decision_programs_model_tools","class_tree_adapter","class_study_grouped_segmentation_checks","--parallel","1")
        run(w/"core/bin/class_study_grouped_segmentation_checks");return
    if a.action=="bind-cohort":
        if not a.dicom_root or not a.out:p.error("bind-cohort requires --dicom-root and --out")
        d=json.loads((ROOT/"protocol/cohort.json").read_text());base=a.dicom_root.resolve()
        for scan in d["scans"]:
            ct=base/scan["expected_ct_series_uid"];seg=base/scan["seg_series_uid"]
            if not ct.is_dir() or not seg.is_dir():raise ValueError(f"Missing series directory {ct} or {seg}")
            candidates=sorted(seg.rglob("*.dcm"))
            if len(candidates)!=1:raise ValueError(f"Require exactly one DICOM SEG file in {seg}; found {len(candidates)}")
            scan["ct_dir"]=str(ct);scan["seg_path"]=str(candidates[0])
        save(a.out.resolve(),d);return
    if a.action=="prepare":
        if not a.manifest or not a.model:p.error("prepare requires --manifest and --model")
        model=a.model.resolve();w.mkdir(parents=True,exist_ok=True)
        expected=json.loads((ROOT/"protocol/checkpoints.json").read_text());receipts=[]
        for i,e in enumerate(expected):
            checkpoint=model/f"fold_{i}/checkpoint_final.pth"
            if sha(checkpoint)!=e["member_sha256"]:raise ValueError(f"Checkpoint identity differs: {checkpoint}")
            receipts.append(e)
        save(w/"point-model-download-complete.json",receipts)
        run(binary,"prepare-segmentation","--manifest",a.manifest.resolve(),"--out",w/"prepared20")
        run(w/"core/bin/class_segmentation_export",w/"prepared20/dataset.json",w/"inputs")
        ll=ROOT/"scripts/lesionlocator"
        run(a.python,ll/"run_reference.py","--cases",w/"inputs/cases.json","--model",model,"--out",w/"reference-point")
        run(a.python,ll/"make_prepared_index.py","--workspace",w)
        run(binary,"prepare-segmentation","--prepared-images",w/"reference-point/prepared-dataset.json","--reference-data",w/"prepared20/dataset.json","--export",w/"inputs/cases.json","--preprocessing-plans",model/"plans.json","--out",w/"normalized20")
        run(a.python,ll/"prepare_cached_mask_channels.py","--workspace",w,"--out",w/"cached-mask")
        run(a.python,ll/"extract_frozen_cnn_channels.py","--workspace",w,"--model",model,"--out",w/"probability-learned32","--check-cached-mask")
        run(a.python,ll/"finalize_channels.py","--workspace",w,"--source",w/"probability-learned32/channels.json","--out",w/"additional-channels54.json")
        run(w/"core/bin/class_segmentation_prepare","--additional-channels",w/"additional-channels54.json","--reference-data",w/"normalized20/dataset.json","--export",w/"inputs/cases.json","--out",w/"prepared54")
        return
    if not a.out:p.error("study/convert requires --out")
    if a.action=="study":
        if not a.library:p.error("study requires --library")
        cmd=[binary,"segmentation-study","--data",(a.data or w/"prepared54/dataset.json").resolve(),"--plan",a.plan.resolve(),"--library",a.library.resolve(),"--out",a.out.resolve()]
        if a.check:cmd.append("--check")
        run(*cmd)
    else:
        if not a.pipeline or not a.library:p.error("convert requires --pipeline and --library")
        cmd=[binary,"convert","--pipeline",a.pipeline.resolve(),"--library",a.library.resolve(),"--out",a.out.resolve()]
        if a.check:cmd.append("--check")
        run(*cmd)
if __name__=="__main__":main()
