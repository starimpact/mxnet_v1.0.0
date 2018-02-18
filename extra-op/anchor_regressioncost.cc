/*!
 * Copyright (c) 2015 by Contributors
 * \file anchor_regressioncost.cc
 * \brief
 * \author Ming Zhang
*/

#include "./anchor_regressioncost-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<cpu>(AnchorRegCostParam param) {
  return new AnchorRegCostOp<cpu>(param);
}

Operator* AnchorRegCostProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}


DMLC_REGISTER_PARAMETER(AnchorRegCostParam);

MXNET_REGISTER_OP_PROPERTY(AnchorRegCost, AnchorRegCostProp)
.add_argument("data", "Symbol", "Input data to the AnchorRegCostOp.")
.add_argument("label", "Symbol", "Label data to the AnchorRegCostOp.")
.add_argument("coordlabel", "Symbol", "CoordLabel data to the AnchorRegCostOp.")
.add_argument("bbslabel", "Symbol", "BBsLabel data to the AnchorRegCostOp.")
.add_argument("infolabel", "Symbol", "AnchorInfoLabel data to the AnchorRegCostOp.")
.add_arguments(AnchorRegCostParam::__FIELDS__())
.describe("Apply AnchorRegCost to input.");

}  // namespace op
}  // namespace mxnet
