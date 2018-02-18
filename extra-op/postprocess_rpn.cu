/*!
 * Copyright (c) 2016 by Contributors
 * \file postprocess_rpn.cu
 * \brief post process of rpn operator
 * \author Ming Zhang
*/
#include "./postprocess_rpn-inl.h"
#include "./mshadow_op.h"
#include <numeric>


namespace mshadow {

namespace cuda {


__global__ void PostProcessRPNForwardKernel(
                int count,
                const float *pfClsAll, const float *pfRegAll, 
                const float *pfAnchor, const float *pfOtherinfo, 
                int dwBatchNum, int dwAnchorNum, int dwFeatH, int dwFeatW, 
                float *pfBBsAll, float *pfScores, int dwMaxBBNum, int *pdwbb_num_all) {
#if 1     
  float clsthreshold = pfOtherinfo[0];
  int originalH = pfOtherinfo[1];
  int originalW = pfOtherinfo[2]; 
//  printf("clsthreshold:%.1f, originalH:%d, originalW:%d\n", clsthreshold, originalH, originalW);
//  __syncthreads();
  int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

  if (1 && index < count)
  {
    int dwFeatSize = dwFeatH * dwFeatW;
    int dwFeatAnchorSize = dwFeatSize * dwAnchorNum;
    int dwBatchI = index / dwFeatAnchorSize;
    int dwAnchorI = (index - dwBatchI * dwFeatAnchorSize) / dwFeatSize;
    int dwRI = (index - dwBatchI * dwFeatAnchorSize - dwFeatSize * dwAnchorI) / dwFeatW;
    int dwCI = (index - dwBatchI * dwFeatAnchorSize - dwFeatSize * dwAnchorI) % dwFeatW;
    int dwOft = dwRI * dwFeatW + dwCI;
    int dwAnchorOft = dwAnchorI * dwFeatSize;
    const float *pfNowAnchor = pfAnchor + dwAnchorI * 2;
    const float *pfReg = pfRegAll + dwFeatAnchorSize * 4 * dwBatchI;
    const float *pfCls = pfClsAll + dwFeatAnchorSize * dwBatchI;
    float *pfBBs = pfBBsAll + dwBatchI * dwMaxBBNum * 5;
    float *pfScs = pfScores + dwBatchI * dwMaxBBNum;
    int *pdwbb_num_now = pdwbb_num_all + dwBatchI;

//    printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d, nownum:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index, nownum);
//    printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index);
//    __syncthreads();
#if 1
    float fScore = pfCls[dwOft + dwAnchorOft];
    if (fScore > clsthreshold)
    {
      int nownum = atomicInc((unsigned int*)(pdwbb_num_now), dwMaxBBNum);
      if (nownum < dwMaxBBNum)
      {
        float fCY = pfReg[dwAnchorOft * 4 + 0 * dwFeatSize + dwOft];
        float fCX = pfReg[dwAnchorOft * 4 + 1 * dwFeatSize + dwOft];
        float fH = pfReg[dwAnchorOft * 4 + 2 * dwFeatSize + dwOft];
        float fW = pfReg[dwAnchorOft * 4 + 3 * dwFeatSize + dwOft];
        fCY = fCY * pfNowAnchor[0] + ((float)(dwRI) * originalH) / dwFeatH;
        fCX = fCX * pfNowAnchor[1] + ((float)(dwCI) * originalW) / dwFeatW;
        fH = expf(fH) * pfNowAnchor[0];
        fW = expf(fW) * pfNowAnchor[1];
       
        {
          pfBBs[nownum * 5 + 0] = fScore;
          pfBBs[nownum * 5 + 1] = fCY;
          pfBBs[nownum * 5 + 2] = fCX;
          pfBBs[nownum * 5 + 3] = fH;
          pfBBs[nownum * 5 + 4] = fW;
          pfScs[nownum] = fScore;
//          printf("bidxx:%d-bidxy:%d-gdimx:%d-bdimx:%d-tidxx:%d, index:%d, nownum:%d\n", blockIdx.x, blockIdx.y, gridDim.x, blockDim.x, threadIdx.x, index, nownum);
//          __syncthreads();
        }
      }
    }
#endif
  }
#endif
}


inline void PostProcessRPNForward(const Tensor<gpu, 4> &datacls_in,
                           const Tensor<gpu, 4> &datareg_in,
                           const Tensor<gpu, 2> &anchorinfo_in,
                           const Tensor<gpu, 1> &otherinfo_in,
                           Tensor<gpu, 3> &bb_out) {
  CHECK_EQ(datacls_in.size(0), datareg_in.size(0));

  int dwBatchNum = datacls_in.size(0);
  int dwAnchorNum = anchorinfo_in.size(0);
  int bb_maxnum_per_batch = bb_out.size(1);
  
  int dwFeatH = datacls_in.size(2);
  int dwFeatW = datacls_in.size(3);

  int dwBufferPerLen = dwAnchorNum * dwFeatH * dwFeatW;

  Stream<gpu>* bbstream = bb_out.stream_;
  Tensor<gpu, 3, float> tBBBuffer(Shape3(dwBatchNum, dwBufferPerLen, 5));
  AllocSpace(&tBBBuffer, false);tBBBuffer.stream_ = bbstream;
  float *pfBBBuffer = tBBBuffer.dptr_;
  int dwBBBufferSize = dwBatchNum * dwBufferPerLen * 5;
  cudaMemset(pfBBBuffer, 0, dwBBBufferSize*sizeof(float));

  Tensor<gpu, 2, float> tScores(Shape2(dwBatchNum, dwBufferPerLen));
  AllocSpace(&tScores, false);tScores.stream_ = bbstream;
  float *pfScores = tScores.dptr_;
  int dwScoreSize = dwBatchNum * dwBufferPerLen;
  cudaMemset(pfScores, 0, dwScoreSize*sizeof(float));

  int dwBBMemLen = bb_out.MSize();
  cudaMemset(bb_out.dptr_, 0, dwBBMemLen*sizeof(float));

  Tensor<gpu, 1, int> tCounter(Shape1(dwBatchNum));
  AllocSpace(&tCounter, false); tCounter.stream_ = bbstream;
  int *pdwCounter = tCounter.dptr_;
  cudaMemset(pdwCounter, 0, dwBatchNum*sizeof(int));

  int count = dwFeatH * dwFeatW * dwAnchorNum * dwBatchNum;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridNum, (gridSize + kMaxGridNum - 1) / kMaxGridNum);
  dim3 dimBlock(kMaxThreadsPerBlock);

  CheckLaunchParam(dimGrid, dimBlock, "PostProcessRPN Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(bb_out.stream_);
  
  PostProcessRPNForwardKernel<<<dimGrid, dimBlock, 0, stream>>>(
            count, 
            datacls_in.dptr_, datareg_in.dptr_, 
            anchorinfo_in.dptr_, otherinfo_in.dptr_, dwBatchNum, dwAnchorNum, dwFeatH, dwFeatW, 
            pfBBBuffer, pfScores, dwBufferPerLen, pdwCounter);
//            bb_out.dptr_, pfScores, bb_maxnum_per_batch, pdwCounter);

#if 1
  {

    Tensor<cpu, 1, float> tRowScore_(Shape1(dwBufferPerLen));AllocSpace(&tRowScore_, false);
    Tensor<cpu, 1, int> tCounter_(Shape1(dwBatchNum));AllocSpace(&tCounter_, false);
    
    Copy<1, int>(tCounter_, tCounter, bbstream);
//    printf("mxnet=>dwCounter[%d, %d]:\n", dwBatchNum, dwBufferPerLen);
    for (int i = 0; i < dwBatchNum; i++)
    {
      Tensor<gpu, 2, float> tRowInfo = tBBBuffer[i];
      Tensor<gpu, 2, float> tRowOut = bb_out[i];
      Tensor<gpu, 1, float> tRowScore = tScores[i];
      Copy<1, float>(tRowScore_, tRowScore, bbstream);
      std::vector<int> index(dwBufferPerLen);
      std::iota(index.begin(), index.end(), 0);
      std::sort(index.begin(), index.end(),
                [&tRowScore_](size_t i0, size_t i1) {return tRowScore_[i0] > tRowScore_[i1];} );
//      printf("batch_%d:%d, \n", i, tCounter_[i]);
      int minnum = std::min(tCounter_[i], bb_maxnum_per_batch);
      for (int j = 0; j < minnum; j++)
      {
//        if (j < 20) printf("%f:%d, ", tRowScore_[index[j]], index[j]);
        Copy(tRowOut[j], tRowInfo[index[j]], bbstream);
      }
//      printf("\n");

   }
   FreeSpace(&tRowScore_);
   FreeSpace(&tCounter_);
  }
#endif
  FreeSpace(&tCounter);
  FreeSpace(&tBBBuffer);
  FreeSpace(&tScores);
  
}
  
} // namespace cuda

inline void PostProcessRPNForward(const Tensor<gpu, 4> &datacls_in,
                           const Tensor<gpu, 4> &datareg_in,
                           const Tensor<gpu, 2> &anchorinfo_in,
                           const Tensor<gpu, 1> &otherinfo_in,
                           Tensor<gpu, 3> &bb_out) {
//  printf("originalW:%d\n", originalW);                           
  cuda::PostProcessRPNForward(datacls_in, datareg_in, anchorinfo_in, otherinfo_in, bb_out);
}

} // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(PostProcessRPNParam param) {
  return new PostProcessRPNOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet
