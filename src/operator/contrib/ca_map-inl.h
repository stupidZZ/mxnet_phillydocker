#ifndef MXNET_OPERATOR_CONTRIB_CA_MAP_INL_H_
#define MXNET_OPERATOR_CONTRIB_CA_MAP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>

#include "../../ndarray/ndarray_function.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

inline bool CAMapOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  CHECK_EQ(in_shape->size(), 2U) << "Input:[weight, value]";
  CHECK_EQ(out_shape->size(), 1U) << "Output:[output]";
  TShape weight_shape(in_shape->at(0));
  TShape value_shape(in_shape->at(1)); 
   
  SHAPE_ASSIGN_CHECK(*out_shape, 0, TShape({value_shape[0], value_shape[1], value_shape[2], value_shape[3]}));
  SHAPE_ASSIGN_CHECK(*in_shape, 0, TShape({out_shape->at(0)[0], out_shape->at(0)[2] + out_shape->at(0)[3]-1, out_shape->at(0)[2], out_shape->at(0)[3]}));
  SHAPE_ASSIGN_CHECK(*in_shape, 1, TShape({out_shape->at(0)[0], out_shape->at(0)[1], out_shape->at(0)[2], out_shape->at(0)[3]}));
  
  return true;
}

inline bool CAMapOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << "Input:[weight, value]";
  CHECK_EQ(out_attrs->size(), 1U) << "Output:[output]";

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}


template<typename xpu, typename DType, typename AccReal>
void CAMapForward(mshadow::Stream<cpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void CAMapBackward(mshadow::Stream<cpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void CAMapForward(mshadow::Stream<gpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output);

template<typename xpu, typename DType, typename AccReal>
void CAMapBackward(mshadow::Stream<gpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output);

template <typename xpu>
inline void CAMapOpForward(const nnvm::NodeAttrs& attrs,
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
    CAMapForward<xpu, DType, AccReal>(s, inputs, outputs);
  });
}


template <typename xpu>
inline void CAMapOpBackward(const nnvm::NodeAttrs& attrs,
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
    CAMapBackward<xpu, DType, AccReal>(s, inputs, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif 
