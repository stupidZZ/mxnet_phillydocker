#ifndef MXNET_OPERATOR_CONTRIB_CA_WEIGHT_INL_H_
#define MXNET_OPERATOR_CONTRIB_CA_WEIGHT_INL_H_

#include <mxnet/operator_util.h>
#include <vector>

#include "../../ndarray/ndarray_function.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct CAWeightParam : public dmlc::Parameter<CAWeightParam> {
  int radius;
  DMLC_DECLARE_PARAMETER(CAWeightParam) {
    DMLC_DECLARE_FIELD(radius).set_range(1, 10000)
    .describe("radius");
  }
};

inline bool CAWeightOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 2U) << "Input:[key, query]";
  CHECK_EQ(out_shape->size(), 1U) << "Output:[weight]";
  TShape key_shape(in_shape->at(0));
  TShape query_shape(in_shape->at(1)); 
   
  if (key_shape[1] == query_shape[1]) {
      SHAPE_ASSIGN_CHECK(*out_shape, 0, TShape({key_shape[0], key_shape[2] + key_shape[3] - 1, key_shape[2], key_shape[3]}));
      SHAPE_ASSIGN_CHECK(*in_shape, 0, TShape({out_shape->at(0)[0], key_shape[1], out_shape->at(0)[2], out_shape->at(0)[3]}));
      SHAPE_ASSIGN_CHECK(*in_shape, 1, TShape({out_shape->at(0)[0], query_shape[1], out_shape->at(0)[2], out_shape->at(0)[3]}));
      return true;
  } else {
      return false;
  }
}

inline bool CAWeightOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << "Input:[key, query]";
  CHECK_EQ(out_attrs->size(), 1U) << "Output:[weight]";

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}


template<typename xpu, typename DType, typename AccReal>
void CAWeightForward(mshadow::Stream<cpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void CAWeightBackward(mshadow::Stream<cpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void CAWeightForward(mshadow::Stream<gpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void CAWeightBackward(mshadow::Stream<gpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output);

template <typename xpu>
inline void CAWeightOpForward(const nnvm::NodeAttrs& attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req[0], kWriteTo);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Fill<false>(s, outputs[0], kWriteTo, 0);
  });
  
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    CAWeightForward<xpu, DType, AccReal>(s, inputs, outputs);
  });
}


template <typename xpu>
inline void CAWeightOpBackward(const nnvm::NodeAttrs& attrs,
                               const OpContext &ctx,
                               const std::vector<TBlob> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<TBlob> &outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req[0], kWriteTo);
  
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  
  // zero grad before backwarding
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Fill<false>(s, outputs[0], kWriteTo, 0);
  });
    
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Fill<false>(s, outputs[1], kWriteTo, 0);
  });
  
  MSHADOW_REAL_TYPE_SWITCH_EX(inputs[0].type_flag_, DType, AccReal, {
    CAWeightBackward<xpu, DType, AccReal>(s, inputs, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif 
