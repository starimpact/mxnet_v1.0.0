/*!
 * Copyright (c) 2015 by Contributors
 * \file anchor_regressioncost.cu
 * \brief
 * \author Ming Zhang
*/

#include "./anchor_regressioncost-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(AnchorRegCostParam param) {
  return new AnchorRegCostOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

