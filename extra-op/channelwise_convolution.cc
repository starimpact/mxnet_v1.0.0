/*!
 * Copyright (c) 2015 by Contributors
 * \file channelwise_convolution.cc
 * \brief
 * \author Yunpeng Chen
*/

#include "./channelwise_convolution-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(ChannelwiseConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ChannelwiseConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ChannelwiseConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

DMLC_REGISTER_PARAMETER(ChannelwiseConvolutionParam);

MXNET_REGISTER_OP_PROPERTY(ChannelwiseConvolution, ChannelwiseConvolutionProp)
.add_argument("data", "Symbol", "Input data to the ChannelwiseConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(ChannelwiseConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

