#include "ca_map-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType, typename AccReal>
void CAMapForward(mshadow::Stream<cpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output) {
  return;
}
                         
template<typename xpu, typename DType, typename AccReal>
void CAMapBackward(mshadow::Stream<cpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output) {
  return;
}
                      
                      

NNVM_REGISTER_OP(_contrib_CAMap)
.describe(R"code(CAMap)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"weight", "value"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", CAMapOpShape)
.set_attr<nnvm::FInferType>("FInferType", CAMapOpType)
.set_attr<FCompute>("FCompute<cpu>", CAMapOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_CAMap"})
.add_argument("weight", "NDArray-or-Symbol", "Input ndarray")
.add_argument("value", "NDArray-or-Symbol", "Input ndarray");

NNVM_REGISTER_OP(_backward_CAMap)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CAMapOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
