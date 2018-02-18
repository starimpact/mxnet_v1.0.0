/*!
 * Copyright (c) 2015 by Contributors
 * \file channelwise_convolution.cu
 * \brief
 * \author Yunpeng Chen
*/

#include "./channelwise_convolution-inl.h"
#include <vector>

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(ChannelwiseConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ChannelwiseConvolutionOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

