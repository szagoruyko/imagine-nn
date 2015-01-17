#include <THC/THC.h>

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit

extern "C"
{
void SpatialAveragePooling_updateOutput(THCudaTensor* input, THCudaTensor* output, int kW, int kH, int dW, int dH);
void SpatialAveragePooling_updateGradInput(THCudaTensor* input, THCudaTensor* gradInput, THCudaTensor* gradOutput, int kW, int kH, int dW, int dH);
}


/*
 * Description:
 *    this function avg-pools an input 3D tensor along dimensions 1 and 2
 *    3D input, 3D output
 */
__global__ void subsample(float *input, float *output, 
                          int input_n, int input_h, int input_w,
                          int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = ceil(float(input_w - kW) / float(dW) + 1);
  int output_h = ceil(float(input_h - kH) / float(dH) + 1);

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*input_w*input_h;

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      // Compute the mean of the input image...
      float *ptr_input = input + yy*dH*input_w + xx*dW;
      float *ptr_output = output + yy*output_w + xx;
      float sum = 0;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          sum += ptr_input[kx];
        ptr_input += input_w; // next input line
      }
      // Update output
      *ptr_output = sum;
    }
  }
}

void SpatialAveragePooling_updateOutput(THCudaTensor* input, THCudaTensor* output, int kW, int kH, int dW, int dH)
{
  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nOutputCols = ceil(float(nInputCols - kW) / float(dW) + 1);
    long nOutputRows = ceil(float(nInputRows - kH) / float(dH) + 1);
    long nInputPlane = input->size[0];

    input = THCudaTensor_newContiguous(input);
    float* input_data = THCudaTensor_data(input);

    THCudaTensor_resize3d(output, nInputPlane, nOutputRows, nOutputCols);
    float* output_data = THCudaTensor_data(output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    subsample <<<blocks, threads>>> (input_data, output_data,
                                     nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nbatch = input->size[0];
    long nOutputCols = ceil(float(nInputCols - kW) / float(dW) + 1);
    long nOutputRows = ceil(float(nInputRows - kH) / float(dH) + 1);
    long nInputPlane = input->size[1];

    input = THCudaTensor_newContiguous(input);
    float* input_data = THCudaTensor_data(input);

    THCudaTensor_resize4d(output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    float* output_data = THCudaTensor_data(output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run subsample kernel
    subsample <<<blocks, threads>>> (input_data, output_data,
                                     nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  }

  // clean
  THCudaTensor_free(input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSubsampling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}


/*
 * Description:
 *    this function computes the gradInput from gradOutput
 */
__global__ void subgradinput(float *gradInput, float *gradOutput, 
                             int input_n, int input_h, int input_w,
                             int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = ceil(float(input_w - kW) / float(dW) + 1);
  int output_h = ceil(float(input_h - kH) / float(dH) + 1);

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float z = *ptr_gradOutput;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++)
          ptr_gradInput[kx] += z;
        ptr_gradInput += input_w;
      }
    }
  }
}


void SpatialAveragePooling_updateGradInput(THCudaTensor* input, THCudaTensor* gradOutput, THCudaTensor* gradInput, int kW, int kH, int dW, int dH)
{
  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];

    float *gradOutput_data = THCudaTensor_data(gradOutput);
    float *gradInput_data;

    THCudaTensor_resizeAs(gradInput, input);
    THCudaTensor_zero(gradInput);
    gradInput_data = THCudaTensor_data(gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    subgradinput <<<blocks, threads>>> (gradInput_data, gradOutput_data,
                                        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];

    float *gradOutput_data = THCudaTensor_data(gradOutput);
    float *gradInput_data;

    THCudaTensor_resizeAs(gradInput, input);
    THCudaTensor_zero(gradInput);
    gradInput_data = THCudaTensor_data(gradInput);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run updateGradInput kernel
    subgradinput <<<blocks, threads>>> (gradInput_data, gradOutput_data, 
                                        nInputPlane, nInputRows, nInputCols, kH, kW, dH, dW);
  }

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialSubsampling.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

#undef CUDA_MAX_THREADS
