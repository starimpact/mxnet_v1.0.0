
#include "./take_op-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(TakeParam_ param) {
  return new TakeOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet
