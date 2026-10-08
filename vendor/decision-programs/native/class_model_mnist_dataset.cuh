#pragma once
// Historical IDX descriptor/loading implementation stays in this adapter.
ModelDataset load_idx_model_dataset(const J&p,const fs::path&out){
 ModelDataset d;
 need(p.at("features")==784&&p.at("classes")==10&&p.at("FIT_rows")==50000&&p.at("VALID_rows")==10000,"MNIST adapter protocol");
 fs::path ip=p.at("images_path").get<std::string>(),lp=p.at("labels_path").get<std::string>();Cpath(ip);Cpath(lp);auto ib=mn_read(ip),lb=mn_read(lp);
 need(sha256(ib)==p.at("images_sha256").get<std::string>()&&sha256(lb)==p.at("labels_sha256").get<std::string>(),"MNIST data pins differ");
 need(ib.size()==47040016&&lb.size()==60008&&be32(ib.data())==2051&&be32(ib.data()+4)==60000&&be32(ib.data()+8)==28&&be32(ib.data()+12)==28&&be32(lb.data())==2049&&be32(lb.data()+4)==60000,"IDX headers");
 d.F=784;d.K=10;d.rows=60000;d.stride=d.F;d.fit_rows=50000;d.valid_rows=10000;
 Dev<uint8_t>raw(d.rows*d.F),rawy(d.rows);cu(cudaMemcpy(raw.p,ib.data()+16,raw.n,cudaMemcpyHostToDevice));cu(cudaMemcpy(rawy.p,lb.data()+8,rawy.n,cudaMemcpyHostToDevice));
 d.x=Dev<float>(d.rows*d.stride);d.labels=Dev<u32>(d.rows);Dev<float>y(d.rows);Dev<u32>ids(d.rows),seen(d.rows),bad(1);seen.zero();bad.zero();Dev<PackStats>ps(1);ps.zero();
 pack<<<blocks(d.rows),256>>>(raw.p,rawy.p,u32(d.rows),u32(d.fit_rows),d.x.p,y.p,ids.p,seen.p,ps.p);audit_pack<<<blocks(d.rows),256>>>(raw.p,rawy.p,u32(d.rows),d.x.p,y.p,ids.p,seen.p,ps.p);class_labels_from_exact_float<<<blocks(d.rows),256>>>(y.p,d.labels.p,d.rows,bad.p);done();need(!ps.at(0).bad&&!bad.at(0),"GPU IDX packing differs");
 auto rowids=ids.get();std::string rowbytes(reinterpret_cast<const char*>(rowids.data()),rowids.size()*4);need(sha256(rowbytes)==p.at("row_ids_sha256").get<std::string>(),"split row IDs differ");if(!out.empty())atomic_text(out/"row-ids.uint32",rowbytes);
 d.binding={{"format","mnist-idx-permutation-1"},{"features",d.F},{"classes",d.K},{"rows",d.rows},{"row_stride",d.stride},{"FIT_rows",d.fit_rows},{"VALID_rows",d.valid_rows},{"images_path",ip.string()},{"images_sha256",sha256(ib)},{"labels_path",lp.string()},{"labels_sha256",sha256(lb)},{"row_ids_sha256",sha256(rowbytes)},{"preprocessing","GPU exact uint8-to-FP32 raw pixel values, no scaling"},{"TEST_read",false}};return d;
}
