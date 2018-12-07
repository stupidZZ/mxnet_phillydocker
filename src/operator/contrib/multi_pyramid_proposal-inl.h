/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file multi_proposal-inl.h
 * \brief MultiProposal Operator
 * \author Piotr Teterwak, Bing Xu, Jian Guo, Xizhou Zhu, Han Hu
*/
#ifndef MXNET_OPERATOR_CONTRIB_MULTI_PYRAMID_PROPOSAL_INL_H_
#define MXNET_OPERATOR_CONTRIB_MULTI_PYRAMID_PROPOSAL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../mshadow_op.h"


namespace mxnet {
namespace op {

namespace proposal {
enum MultiPyramidProposalOpInputs {kImInfo, kClsProbStride4, kBBoxPredStride4, kClsProbStride8, kBBoxPredStride8, kClsProbStride16, kBBoxPredStride16, kClsProbStride32, kBBoxPredStride32, kClsProbStride64, kBBoxPredStride64};
enum MultiPyramidProposalOpOutputs {kOut, kScore};
enum MultiPyramidProposalForwardResource {kTempResource};
}  // proposal

struct MultiPyramidProposalParam : public dmlc::Parameter<MultiPyramidProposalParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  int rpn_min_size;
  nnvm::Tuple<float> scales;
  nnvm::Tuple<float> ratios;
  nnvm::Tuple<int> feature_stride;
  bool output_score;
  bool iou_loss;
  DMLC_DECLARE_PARAMETER(MultiPyramidProposalParam) {
    float tmp[] = {0, 0, 0, 0};
    int tmp_int[] = {0, 0, 0, 0, 0};
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(16)
    .describe("Minimum height or width in proposal");
    tmp[0] = 4.0f; tmp[1] = 8.0f; tmp[2] = 16.0f; tmp[3] = 32.0f;
    DMLC_DECLARE_FIELD(scales).set_default(nnvm::Tuple<float>(tmp, tmp + 4))
    .describe("Used to generate anchor windows by enumerating scales");
    tmp[0] = 0.5f; tmp[1] = 1.0f; tmp[2] = 2.0f;
    DMLC_DECLARE_FIELD(ratios).set_default(nnvm::Tuple<float>(tmp, tmp + 3))
    .describe("Used to generate anchor windows by enumerating ratios");
    tmp_int[0] = 4; tmp_int[1] = 8; tmp_int[2] = 16; tmp_int[3] = 32; tmp_int[4] = 64;
    DMLC_DECLARE_FIELD(feature_stride).set_default(nnvm::Tuple<int>(tmp_int, tmp_int + 5))
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
    DMLC_DECLARE_FIELD(output_score).set_default(false)
    .describe("Add score to outputs");
    DMLC_DECLARE_FIELD(iou_loss).set_default(false)
    .describe("Usage of IoU Loss");
  }
};

template<typename xpu>
Operator *CreateOp(MultiPyramidProposalParam param);

#if DMLC_USE_CXX11
class MultiPyramidProposalProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), param_.feature_stride.ndim() * 2 + 1) << "Input:[cls_prob, bbox_pred, im_info]";
    const TShape &dshape = in_shape->at(proposal::kClsProbStride4);
    if (dshape.ndim() == 0) return false;
    Shape<4> bbox_pred_shape;
    bbox_pred_shape = Shape4(dshape[0], dshape[1] * 2, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kBBoxPredStride4,
                       bbox_pred_shape);
    Shape<2> im_info_shape;
    im_info_shape = Shape2(dshape[0], 3);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kImInfo, im_info_shape);
    out_shape->clear();
    // output
    out_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 5));
    // score
    out_shape->push_back(Shape2(dshape[0] * param_.rpn_post_nms_top_n, 1));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new MultiPyramidProposalProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_MultiPyramidProposal";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_score) {
      return 2;
    } else {
      return 1;
    }
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    std::vector<std::string> args;
    std::string s = "im_info";
    args.push_back(s);
    for (int i = 0; i < param_.feature_stride.ndim(); ++i){
        std::string s = "rpn_cls_prob_stride" + std::to_string(param_.feature_stride[i]);
        args.push_back(s);
        std::string s2 = "rpn_bbox_pred_stride" + std::to_string(param_.feature_stride[i]);
        args.push_back(s2);
    }
    return args;
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "score"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  MultiPyramidProposalParam param_;
};  // class MultiProposalProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet

//========================
// Anchor Generation Utils
//========================
namespace mxnet {
namespace op {
namespace utils {

inline void _MakeAnchor(float w,
                        float h,
                        float x_ctr,
                        float y_ctr,
                        std::vector<float> *out_anchors) {
  out_anchors->push_back(x_ctr - 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr - 0.5f * (h - 1.0f));
  out_anchors->push_back(x_ctr + 0.5f * (w - 1.0f));
  out_anchors->push_back(y_ctr + 0.5f * (h - 1.0f));
  out_anchors->push_back(0.0f);
}

inline void _Transform(float scale,
                       float ratio,
                       const std::vector<float>& base_anchor,
                       std::vector<float>  *out_anchors) {
  float w = base_anchor[2] - base_anchor[1] + 1.0f;
  float h = base_anchor[3] - base_anchor[1] + 1.0f;
  float x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  float y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = std::floor(std::sqrt(size_ratios) + 0.5f) * scale;
  float new_h = std::floor((new_w / scale * ratio) + 0.5f) * scale;

  _MakeAnchor(new_w, new_h, x_ctr,
             y_ctr, out_anchors);
}

// out_anchors must have shape (n, 5), where n is ratios.size() * scales.size()
inline void GenerateAnchors(const std::vector<float>& base_anchor,
                            const nnvm::Tuple<float>& ratios,
                            const nnvm::Tuple<float>& scales,
                            std::vector<float> *out_anchors) {
  for (size_t j = 0; j < ratios.ndim(); ++j) {
    for (size_t k = 0; k < scales.ndim(); ++k) {
      _Transform(scales[k], ratios[j], base_anchor, out_anchors);
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

#endif  //  MXNET_OPERATOR_CONTRIB_MULTI_PROPOSAL_INL_H_
