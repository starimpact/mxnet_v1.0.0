/*!
 * Copyright (c) 2017 by MingZhang
 * \file take_op-inl.h
 * \brief take element from a tensor.
*/
#ifndef MXNET_OPERATOR_TAKE_INL_H_
#define MXNET_OPERATOR_TAKE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace take_ {
enum TakeOpInputs {kData, kIndex};
enum TakeOpOutputs {kOut};
enum TakeOpResource {kTempSpace};
}  //namespace take_

struct TakeParam_ : public dmlc::Parameter<TakeParam_> {
  DMLC_DECLARE_PARAMETER(TakeParam_) {}
};

template<typename xpu>
class TakeOp : public Operator {
 public:
  explicit TakeOp(TakeParam_ p) : param_(p) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &qux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
//    Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    TShape dshape = in_data[take_::kData].shape_;
//    std::cout << "take_" << dshape[0] << dshape[1] << "\n";
    Tensor<xpu, 3> data = in_data[take_::kData].get_with_shape<xpu, 3, real_t>(Shape3(dshape[0], dshape[1], 1), s); 
    Tensor<xpu, 1> index = in_data[take_::kIndex].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> out = out_data[take_::kOut].get_with_shape<xpu, 2, real_t>(Shape2(dshape[0], 1), s);
    size_t size0 = index.size(0);
    Tensor<cpu, 1> index_cpu = NewTensor<cpu>(Shape1(size0), 0.f);
//    Tensor<cpu, 1> index_cpu = ctx.requested[take_::kTempSpace].get_space<cpu>(Shape1(1), s_cpu);
    Copy<1, real_t>(index_cpu, index, s);
    for (size_t idx = 0; idx < size0; idx++) { 
      int idx0 = static_cast<int>(index_cpu[idx]);
      Copy<1, real_t>(out[idx], data[idx][idx0], s);
//      out[idx] = data[idx][idx0];
    }
    FreeSpace(&index_cpu);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
//    Stream<cpu> *s_cpu = ctx.get_stream<cpu>();
    Tensor<xpu, 1> index = in_data[take_::kIndex].get<xpu, 1, real_t>(s);
    TShape ishape = out_grad[take_::kOut].shape_;
    Tensor<xpu, 2> grad_out = out_grad[take_::kOut].get_with_shape<xpu, 2, real_t>(Shape2(ishape[0], 1), s);
    TShape ishape2 = in_grad[take_::kData].shape_;
    Tensor<xpu, 3> grad_in = in_grad[take_::kData].get_with_shape<xpu, 3, real_t>(Shape3(ishape2[0], ishape2[1], 1), s);
    size_t size0 = index.size(0);
    Tensor<cpu, 1> index_cpu = NewTensor<cpu>(Shape1(size0), 0.f);
//    Tensor<cpu, 1> index_cpu = ctx.requested[take_::kTempSpace].get_space<cpu>(Shape1(1), s_cpu);
    Copy<1, real_t>(index_cpu, index, s);
    if (req[take_::kOut] == kWriteTo) {
      grad_in = 0.f;
    }
    for (size_t idx = 0; idx < size0; idx++) {
      int idx0 = static_cast<int>(index_cpu[idx]);
      grad_in[idx][idx0] += grad_out[idx];
    }
    FreeSpace(&index_cpu);
  }

 private:
  TakeParam_ param_;
};  // class TakeOp

template<typename xpu>
Operator* CreateOp(TakeParam_ param);


#if DMLC_USE_CXX11
class TakeProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "index"};
  }
  
  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Take operator must have 2 inputs.";
    const TShape &dshape = (*in_shape)[take_::kData];
    const TShape &ishape = (*in_shape)[take_::kIndex];
    CHECK_EQ(dshape.ndim(), 2) << "data shape must be 2 dimension.";
    CHECK_EQ(ishape.ndim(), 1) << "index shape must be 1 dimension.";
    CHECK_GE(ishape[0], 1) << "index must be a scalar or a vector.";
    TShape oshape = Shape2(dshape[0], 1);
    out_shape->clear();
    out_shape->push_back(oshape);

    return true;
  }

  OperatorProperty* Copy() const override {
    TakeProp* sym = new TakeProp();
    sym->param_ = this->param_;
    return sym;
  }

  std::string TypeString() const override {
    return "Take";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[take_::kOut], in_data[take_::kIndex]};
  }

//  std::vector<ResourceRequest> ForwardResource(
//      const std::vector<TShape> &in_shape) const override {
//    return {ResourceRequest::kTempSpace};
//  }
//
//  std::vector<ResourceRequest> BackwardResource(
//      const std::vector<TShape> &in_shape) const override {
//    return {ResourceRequest::kTempSpace};
//  }

  Operator* CreateOperator(Context ctx) const override;
 private:
  TakeParam_ param_;
};  // class TakeProp
#endif

}  // namespace op
}  // namespace mxnet

#endif
