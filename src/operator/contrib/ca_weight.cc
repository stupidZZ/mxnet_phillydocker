#include "ca_weight-inl.h"

namespace mxnet {
namespace op {

template<typename xpu, typename DType, typename AccReal>
void CAWeightForward(mshadow::Stream<cpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output) {
  return;
}
                         
template<typename xpu, typename DType, typename AccReal>
void CAWeightBackward(mshadow::Stream<cpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output) {
  return;
}
                      
                      
DMLC_REGISTER_PARAMETER(CAWeightParam);

NNVM_REGISTER_OP(_contrib_CAWeight)
.describe(R"code(CAWeight)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CAWeightParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"key", "query"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", CAWeightOpShape)
.set_attr<nnvm::FInferType>("FInferType", CAWeightOpType)
.set_attr<FCompute>("FCompute<cpu>", CAWeightOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_CAWeight"})
.add_argument("key", "NDArray-or-Symbol", "Input ndarray")
.add_argument("query", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(CAWeightParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_CAWeight)
.set_attr_parser(ParamParser<CAWeightParam>)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", CAWeightOpBackward<cpu>);

}  // namespace op
}  // namespace mxnet
