#include <cuda_runtime_api.h>
#include <algorithm>
#include "ca_weight-inl.h"


namespace mxnet {
namespace op {

using namespace mshadow;

template<typename xpu, typename DType, typename AccReal>
__global__ void ca_forward_kernel(const Tensor<xpu, 4, DType> t, const Tensor<xpu, 4, DType> f, Tensor<xpu, 4, DType> weight, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int sp = height * width;
  //int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < height+width-1) {
    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        //DType _t = t[(batch * chn + plane) * sp + y*width + x];
        DType _t = t[batch][plane][y][x];
        if (z < width) {
          int i = z;
          //DType _f = f[(batch * chn + plane) * sp + y*width + i];
          DType _f = f[batch][plane][y][i];
          //weight[(batch * len + i) * sp + y*width + x] += _t*_f;
          weight[batch][i][y][x] += _t*_f;
        } else {
          int i = z - width;
          int j = i<y ? i : i+1;

          //DType _f = f[(batch * chn + plane) * sp + j*width + x];
          DType _f = f[batch][plane][j][x];
          //weight[(batch * len + width + i) * sp + y*width + x] += _t*_f;
          weight[batch][width+i][y][x] += _t*_f;
        }
      }
    }
  }
}

template<typename xpu, typename DType, typename AccReal>
__global__ void ca_backward_kernel_t(const Tensor<xpu, 4, DType> dw, const Tensor<xpu, 4, DType> t, const Tensor<xpu, 4, DType> f, Tensor<xpu, 4, DType> dt,
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int sp = height * width;
  //int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {
        
        for (int i = 0; i < width; ++i) {
          //DType _dw = dw[(batch * len + i) * sp + y*width + x];
          //DType _f = f[(batch * chn + plane) * sp + y*width + i];
          //dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;

          DType _dw = dw[batch][i][y][x];
          DType _f = f[batch][plane][y][i];
          dt[batch][plane][y][x] += _dw * _f;
        }
        for (int i = 0; i < height; ++i)  {
          if (i == y) continue;
          int j = i<y ? i : i-1;

          //DType _dw = dw[(batch * len + width + j) * sp + y*width + x];
          //DType _f = f[(batch * chn + plane) * sp + i*width + x];
          //dt[(batch * chn + plane) * sp + y*width + x] += _dw * _f;
          DType _dw = dw[batch][width+j][y][x];
          DType _f = f[batch][plane][i][x];
          dt[batch][plane][y][x] += _dw * _f;
        }
    }
  }
}

template<typename xpu, typename DType, typename AccReal>
__global__ void ca_backward_kernel_f(const Tensor<xpu, 4, DType> dw, const Tensor<xpu, 4, DType> t, const Tensor<xpu, 4, DType> f, Tensor<xpu, 4, DType> df, 
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int sp = height * width;
  //int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    
    for (int batch = 0; batch < num; ++batch) {
      
      for (int i = 0; i < width; ++i) {
        //DType _dw = dw[(batch * len + x) * sp + y*width + i];
        //DType _t = t[(batch * chn + plane) * sp + y*width + i];
        //df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
        DType _dw = dw[batch][x][y][i];
        DType _t = t[batch][plane][y][i];
        df[batch][plane][y][x] += _dw * _t;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;
        int j = i>y ? y : y-1;

        //DType _dw = dw[(batch * len + width + j) * sp + i*width + x];
        //DType _t = t[(batch * chn + plane) * sp + i*width + x];
        //df[(batch * chn + plane) * sp + y*width + x] += _dw * _t;
        DType _dw = dw[batch][width+j][i][x];
        DType _t = t[batch][plane][i][x];
        df[batch][plane][y][x] += _dw * _t;
      }
    }

  }
}

template<typename xpu, typename DType, typename AccReal>
void CAWeightForward(mshadow::Stream<gpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output) {
  
  Tensor<xpu, 4, DType> key_data = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> query_data = input[1].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> output_data = output[0].get<xpu, 4, DType>(s);
  int output_height = output_data.size(2);
  int output_width = output_data.size(3);
  int output_dim = output_data.size(1);
  
  int input_batch = key_data.size(0);
  int input_dim = key_data.size(1);
  int input_height = key_data.size(2);
  int input_width = key_data.size(3);
  
  //printf("intput_dim:%d\n", input_dim);
  dim3 threads(32, 32);
  int d1 = (input_width+threads.x-1)/threads.x;
  int d2 = (input_height+threads.y-1)/threads.y;
  int d3 = input_width + input_height - 1;
  //printf("d1:%d, d2:%d, d3:%d\n", d1, d2, d3);
  dim3 blocks(d1, d2, d3);
  CHECK_EQ(d3, output_dim);

  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  ca_forward_kernel<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(query_data, key_data, output_data, input_batch, input_dim, input_height, input_width);
  
  MSHADOW_CUDA_POST_KERNEL_CHECK(ca_forward_kernel);
}

template<typename xpu, typename DType, typename AccReal>
void CAWeightBackward(mshadow::Stream<gpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> out_grad = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> key_data = input[1].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> query_data = input[2].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> key_grad = output[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> query_grad = output[1].get<xpu, 4, DType>(s);
  
  int output_height = out_grad.size(2);
  int output_width = out_grad.size(3);
  int output_dim = out_grad.size(1);
  
  int input_batch = key_data.size(0);
  int input_dim = key_data.size(1);
  int input_height = key_data.size(2);
  int input_width = key_data.size(3);
  //printf("output_dim:%d input_dim:%d\n", output_dim, input_dim);

  dim3 threads(32, 32);
  int d1 = (input_width+threads.x-1)/threads.x;
  int d2 = (input_height+threads.y-1)/threads.y;
  int d3 = input_dim;
  dim3 blocks(d1, d2, d3);
  
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  ca_backward_kernel_t<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(out_grad, query_data, key_data, query_grad, input_batch, input_dim, input_height, input_width);
  ca_backward_kernel_f<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(out_grad, query_data, key_data, key_grad, input_batch, input_dim, input_height, input_width);
  
  MSHADOW_CUDA_POST_KERNEL_CHECK(ca_backward_kernel_t);
  MSHADOW_CUDA_POST_KERNEL_CHECK(ca_backward_kernel_f);
}

NNVM_REGISTER_OP(_contrib_CAWeight)
.set_attr<FCompute>("FCompute<gpu>", CAWeightOpForward<gpu>);

NNVM_REGISTER_OP(_backward_CAWeight)
.set_attr<FCompute>("FCompute<gpu>", CAWeightOpBackward<gpu>);

}
}