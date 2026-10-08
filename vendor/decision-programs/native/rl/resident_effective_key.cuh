#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
namespace rl_qualified_session::resident_detail {
using U=std::uint64_t;
__device__ void effective_key_node(const U*source,U nn,const unsigned*order,
    bool injected,U*out,int*error,U id){
  if(*error)return;

  if(id==0){out[0]=source[0];out[1]=nn;}
  if(id>=nn)return;U*key=out+2+45*id;
  key[0]=id;key[1]=12;key[2]=UINT64_MAX;key[3]=0;key[4]=0;
  for(unsigned k=0;k<40;++k)key[5+k]=UINT_MAX;
  const U*node=source+37+7*id;U kind=node[0],dim=node[1],first=node[4],count=node[5];
  if(!kind||dim<10)return;
  U na=source[23];if(kind!=1||dim>=12||count<2||first>na||count>na-first){atomicCAS(error,0,5);return;}
  const U*arcs=source+37+7*nn;U fallback=0,seen=0;int largest=-1;
  U group=dim==10?15:((U(1)<<44)-1)^U(15),allowed=source[21]&group;
  for(U j=0;j<count;++j){const U*arc=arcs+4*(first+j);U mask=arc[2];
    if(!mask||(mask&~allowed)||(mask&seen)||arc[3]>=id){atomicCAS(error,0,6);return;}seen|=mask;
    int size=__popcll(mask);if(size>largest){largest=size;fallback=j;}
  }
  if(seen!=allowed){atomicCAS(error,0,7);return;}
  U selected=allowed&~arcs[4*(first+fallback)+2];unsigned at=0;
  key[1]=dim;key[2]=fallback;key[3]=selected;key[4]=__popcll(selected);
  if(injected){unsigned begin=dim==10?0:4,end=dim==10?4:44;
    for(unsigned k=begin;k<end;++k){if(order[k]>=44){atomicCAS(error,0,3);return;}if(selected&(U(1)<<order[k]))key[5+at++]=order[k];}
  }else{
    // Default lower_build emits arcs descending and bits ascending. Its root
    // chain therefore visits arcs ascending and bits descending within each.
    for(U j=0;j<count;++j)if(j!=fallback){U mask=arcs[4*(first+j)+2];
      while(mask){unsigned bit=63-__clzll(mask);mask^=U(1)<<bit;key[5+at++]=bit;}
    }
  }
  if(at!=key[4]||at>40)atomicCAS(error,0,8);
}
}
