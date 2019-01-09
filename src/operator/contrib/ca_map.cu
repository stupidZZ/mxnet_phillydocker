#include <cuda_runtime_api.h>
#include <algorithm>
#include "ca_map-inl.h"


namespace mxnet {
namespace op {

using namespace mshadow;

template<typename xpu, typename DType, typename AccReal>
__global__ void ca_map_forward_kernel(const Tensor<xpu, 4, DType> weight, const Tensor<xpu, 4, DType> g, Tensor<xpu, 4, DType> out, int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int sp = height * width;
  //int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {
    for (int batch = 0; batch < num; ++batch) {

      for (int i = 0; i < width; ++i) {
        //float _g = g[(batch * chn + plane) * sp + y*width + i];
        //float _w = weight[(batch * len + i) * sp + y*width + x];
        //out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
        DType _g = g[batch][plane][y][i];
        DType _w = weight[batch][i][y][x];
        out[batch][plane][y][x] += _g * _w;
      }
      for (int i = 0; i < height; ++i) {
        if (i == y) continue;

        int j = i<y ? i : i-1;

        //float _g = g[(batch * chn + plane) * sp + i*width + x];
        //float _w = weight[(batch * len + width + j) * sp + y*width + x];
        //out[(batch * chn + plane) * sp + y*width + x] += _g * _w;
        DType _g = g[batch][plane][i][x];
        DType _w = weight[batch][width+j][y][x];
        out[batch][plane][y][x] += _g * _w;
      }
    }
  }

}

template<typename xpu, typename DType, typename AccReal>
__global__ void ca_map_backward_kernel_w(const Tensor<xpu, 4, DType> dout, const Tensor<xpu, 4, DType> weight, const Tensor<xpu, 4, DType> g, Tensor<xpu, 4, DType> dw,
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int sp = height * width;
  //int len = height + width - 1;
  int z = blockIdx.z;

  if (x < width && y < height && z < height+width-1) {

    for (int batch = 0; batch < num; ++batch) {
      for (int plane = 0; plane < chn; ++plane) {
        //float _dout = dout[(batch * chn + plane) * sp + y*width + x];
        DType _dout = dout[batch][plane][y][x];
        if (z < width) {
          int i = z;
          //float _g = g[(batch * chn + plane) * sp + y*width + i];
          //dw[(batch * len + i) * sp + y*width + x] += _dout * _g;
          DType _g = g[batch][plane][y][i];
          dw[batch][i][y][x] += _dout * _g;
        } else {
          int i = z - width;
          int j = i<y ? i : i+1;

          //float _g = g[(batch * chn + plane) * sp + j*width + x];
          //dw[(batch * len + width + i) * sp + y*width + x] += _dout * _g;
          DType _g = g[batch][plane][j][x];
          dw[batch][width+i][y][x] += _dout * _g;
        }
      }
    }
  }
}

template<typename xpu, typename DType, typename AccReal>
__global__ void ca_map_backward_kernel_g(const Tensor<xpu, 4, DType> dout, const Tensor<xpu, 4, DType> weight, const Tensor<xpu, 4, DType> g, Tensor<xpu, 4, DType> dg, 
                                int num, int chn, int height, int width) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  //int sp = height * width;
  //int len = height + width - 1;
  int plane = blockIdx.z;

  if (x < width && y < height && plane < chn) {

    for (int batch = 0; batch < num; ++batch) {
      for (int i = 0; i < width; ++i) {
        //float _dout = dout[(batch * chn + plane) * sp + y*width + i];
        //float _w = weight[(batch * len + x) * sp + y*width + i];
        //dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
        DType _dout = dout[batch][plane][y][i];
        DType _w = weight[batch][x][y][i];
        dg[batch][plane][y][x] += _dout * _w;
      }

      for (int i = 0; i < height; ++i) {
        if (i == y) continue;
        int j = i>y ? y : y-1;

        //float _dout = dout[(batch * chn + plane) * sp + i*width + x];
        //float _w = weight[(batch * len + width + j) * sp + i*width + x];
        //dg[(batch * chn + plane) * sp + y*width + x] += _dout * _w;
        DType _dout = dout[batch][plane][i][x];
        DType _w = weight[batch][width+j][i][x];
        dg[batch][plane][y][x] += _dout * _w; 
      }
    }
  }
}


template<typename xpu, typename DType, typename AccReal>
void CAMapForward(mshadow::Stream<gpu> *s,
                     const std::vector<TBlob> &input,
                     const std::vector<TBlob> &output) {
  
  Tensor<xpu, 4, DType> weight_data = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> value_data = input[1].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> output_data = output[0].get<xpu, 4, DType>(s);
  int output_height = output_data.size(2);
  int output_width = output_data.size(3);
  int output_dim = output_data.size(1);
  
  int value_batch = value_data.size(0);
  int value_dim = value_data.size(1);
  int value_height = value_data.size(2);
  int value_width = value_data.size(3);
  
  //printf("intput_dim:%d\n", input_dim);
  dim3 threads(32, 32);
  int d1 = (value_width+threads.x-1)/threads.x;
  int d2 = (value_height+threads.y-1)/threads.y;
  int d3 = value_dim;
  //printf("d1:%d, d2:%d, d3:%d\n", d1, d2, d3);
  dim3 blocks(d1, d2, d3);
  CHECK_EQ(d3, output_dim);

  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  ca_map_forward_kernel<xpu, DType, AccReal><<<blocks, threads, 0, stream>>>(weight_data, value_data, output_data, value_batch, value_dim, value_height, value_width);
  
  MSHADOW_CUDA_POST_KERNEL_CHECK(ca_map_forward_kernel);
}

template<typename xpu, typename DType, typename AccReal>
void CAMapBackward(mshadow::Stream<gpu> *s,
                      const std::vector<TBlob> &input,
                      const std::vector<TBlob> &output) {
  Tensor<xpu, 4, DType> out_grad = input[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> weight_data = input[1].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> value_data = input[2].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> weight_grad = output[0].get<xpu, 4, DType>(s);
  Tensor<xpu, 4, DType> value_grad = output[1].get<xpu, 4, DType>(s);
  
  int output_height = out_grad.size(2);
  int output_width = out_grad.size(3);
  int output_dim = out_grad.size(1);
  
  int value_batch = value_data.size(0);
  int value_dim = value_data.size(1);
  int value_height = value_data.size(2);
  int value_width = value_data.size(3);
  //printf("output_dim:%d input_dim:%d\n", output_dim, input_dim);

  dim3 threads(32, 32);
  int d1 = (value_width+threads.x-1)/threads.x;
  int d2 = (value_height+threads.y-1)/threads.y;
  dim3 blocks_1(d1, d2, value_width + value_height - 1);
  
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  ca_map_backward_kernel_w<xpu, DType, AccReal><<<blocks_1, threads, 0, stream>>>(out_grad, weight_data, value_data, weight_grad, value_batch, value_dim, value_height, value_width);
  
  dim3 blocks_2(d1, d2, value_dim);
  ca_map_backward_kernel_g<xpu, DType, AccReal><<<blocks_2, threads, 0, stream>>>(out_grad, weight_data, value_data, value_grad, value_batch, value_dim, value_height, value_width);
  
  MSHADOW_CUDA_POST_KERNEL_CHECK(ca_map_backward_kernel_w);
  MSHADOW_CUDA_POST_KERNEL_CHECK(ca_map_backward_kernel_g);
}

NNVM_REGISTER_OP(_contrib_CAMap)
.set_attr<FCompute>("FCompute<gpu>", CAMapOpForward<gpu>);

NNVM_REGISTER_OP(_backward_CAMap)
.set_attr<FCompute>("FCompute<gpu>", CAMapOpBackward<gpu>);

}
}