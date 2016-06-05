// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

// Torch port:
// IMAGINE, Sergey Zagoruyko, Francisco Massa, 2015

#include "THC.h"
#include <algorithm>
#include <cfloat>
#include "assert.h"

#include "common.h"

#define NUM_BUFFERS 6
#define PRECISION_LIMIT 1e-10
#define MIN_BIN_SIZE 2.0f

using std::max;
using std::min;

template <typename Dtype>
__global__ void ROIWarpBilinearSampleForward(
    const int nthreads, const Dtype* bottom_data,
    const int channels, 
    const int height, const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_grid_ctrs, const Dtype* bottom_bin_sizes, const Dtype* bottom_roi_batch_inds,  
    Dtype* top_data, 
    Dtype* top_data_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int roi_batch_ind = bottom_roi_batch_inds[n] - 1;
    int grid_ctr_ind = n * (pooled_height * pooled_width * 2) +  ph * (pooled_width * 2) + pw * 2;
    int bin_size_ind = n * 2;
 
    Dtype wctr = bottom_grid_ctrs[grid_ctr_ind+0] - 1;
    Dtype hctr = bottom_grid_ctrs[grid_ctr_ind+1] - 1;

    Dtype bin_size_w = max(bottom_bin_sizes[bin_size_ind+0], MIN_BIN_SIZE);
    Dtype bin_size_h = max(bottom_bin_sizes[bin_size_ind+1], MIN_BIN_SIZE);
 
    Dtype wstart_ = wctr - bin_size_w / 2.0;
    Dtype hstart_ = hctr - bin_size_h / 2.0;
    Dtype wend_   = wctr + bin_size_w / 2.0;
    Dtype hend_   = hctr + bin_size_h / 2.0;
              
    int wstart = static_cast<int>(floor(wstart_)); 
    int hstart = static_cast<int>(floor(hstart_)); 
    int wend   = static_cast<int>( ceil(wend_));// + 1;
    int hend   = static_cast<int>( ceil(hend_));// + 1;
 
    //top_data[index] = hend+1;
    //top_data[index] = wend+1;
    //top_data[index] = hstart+1;
    //top_data[index] = wstart+1;
    //top_data[index] = wstart_+1;
    //top_data[index] = wend_+1;
    //top_data[index] = hctr+1;
    //top_data[index] = wctr+1;
    //top_data[index] = bin_size_w;
    //top_data[index] = bin_size_h;
    //top_data[index] = roi_batch_ind + 1; 
   
    //// Add roi offsets and clip to input boundaries
    //hstart = min(max(hstart, 0), height);
    //hend   = min(max(hend, 0),   height);
    //wstart = min(max(wstart, 0), width );
    //wend   = min(max(wend, 0),   width );

    //top_data[index] = hstart+1; 
    //top_data[index] = wstart+1;
    //top_data[index] = hend+1;
    //top_data[index] = wend+1;

    // Auxilliary variables used in backprop 
    Dtype w_mask = 0, h_mask = 0; 
    Dtype dgx_final_dwctr_all  = 0;
    Dtype dgx_final_dwdiff_all = 0;
    Dtype dgy_final_dhctr_all  = 0;
    Dtype dgy_final_dhdiff_all = 0;
 
    // Define an empty pooling region to be zero
    Dtype val = 0; Dtype gain = 0, gain_x = 0, gain_y = 0, gain_x_all = 0, gain_y_all = 0;   
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h <= hend; ++h) {
      Dtype h_ = h;
      h_mask = ((hstart_ <= h_ && h_ <= hend_) ? 1.0 : 0); 
      for (int w = wstart; w <= wend; ++w) {
        int bottom_index = h * width + w;
        Dtype w_ = w;  
        w_mask = ((wstart_ <= w_ && w_ <= wend_) ? 1.0 : 0);
 
        //gain_x = (bin_size_w+1) - abs(w_ - wctr);
        //gain_y = (bin_size_h+1) - abs(h_ - hctr);
        gain_x = w_mask * (bin_size_w - abs(w_ - wctr));
        gain_y = h_mask * (bin_size_h - abs(h_ - hctr));

        gain = gain_x * gain_y;
 
        if (0 <= h && h < height && 0 <= w && w < width) { 
          val = val + gain * bottom_data[bottom_index];
          //val = val + gain; // for debug
        }
        //val = val + gain; // for debug

        if (h == hstart) {
          gain_x_all = gain_x_all + gain_x;

          // Update information used in backprop
          dgx_final_dwctr_all  = dgx_final_dwctr_all  + w_mask * (w_ >= wctr ? 1 : -1);
          dgx_final_dwdiff_all = dgx_final_dwdiff_all + w_mask;
        }
      }
      gain_y_all = gain_y_all + gain_y;
        
      dgy_final_dhctr_all  = dgy_final_dhctr_all  + h_mask * (h >= hctr ? 1 : -1);
      dgy_final_dhdiff_all = dgy_final_dhdiff_all + h_mask;
    }
    if (gain_x_all > PRECISION_LIMIT)
      val = val / gain_x_all;
    if (gain_y_all > PRECISION_LIMIT)  
      val = val / gain_y_all;
    top_data[index] = val;

    //top_data[index] = gain_y_all; // for debug

    if (c == 0) { 
      int buffer_index = n * (pooled_height * pooled_width * NUM_BUFFERS) + ph * (pooled_width * NUM_BUFFERS) + pw * NUM_BUFFERS;
      top_data_buffer[buffer_index+0] = gain_x_all; 
      top_data_buffer[buffer_index+1] = gain_y_all;
      top_data_buffer[buffer_index+2] = dgx_final_dwctr_all;
      top_data_buffer[buffer_index+3] = dgy_final_dhctr_all;
      top_data_buffer[buffer_index+4] = dgx_final_dwdiff_all;
      top_data_buffer[buffer_index+5] = dgy_final_dhdiff_all;
    }
  }
}

extern "C"
void inn_ROIWarpingBilinearSample_updateOutput(THCState *state,
    THCudaTensor *output, THCudaTensor *output_buffer, 
    THCudaTensor *data, THCudaTensor* grid_ctrs, THCudaTensor* bin_sizes, THCudaTensor* roi_batch_inds,
    int width, int height)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, grid_ctrs) == 4 && grid_ctrs->size[3] == 2);
  THAssert(THCudaTensor_nDimension(state, bin_sizes) == 2 && bin_sizes->size[1] == 2);
  THAssert(THCudaTensor_nDimension(state, roi_batch_inds) == 2 
           && roi_batch_inds->size[0] == grid_ctrs->size[0] 
           && roi_batch_inds->size[0] == bin_sizes->size[0]);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, grid_ctrs));
  THAssert(THCudaTensor_isContiguous(state, bin_sizes));

  long nInputPlane = data->size[1];

  // update output
  long count = THCudaTensor_nElement(state, output);
  ROIWarpBilinearSampleForward<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      THCudaTensor_data(state, data),
      nInputPlane, data->size[2], data->size[3], height, width,
      THCudaTensor_data(state, grid_ctrs),
      THCudaTensor_data(state, bin_sizes),
      THCudaTensor_data(state, roi_batch_inds),
      THCudaTensor_data(state, output), 
      THCudaTensor_data(state, output_buffer) 
      );

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inn_ROIWarpingBilinearSample_updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

template <typename Dtype>
__global__ void ROIWarpBilinearBackwardData(
    const int nthreads, 
    const int channels, const int height, const int width, const int pooled_height, const int pooled_width, 
    const int nth_roi,
    const Dtype* bottom_grid_ctrs, 
    const Dtype* bottom_bin_sizes, 
    const Dtype* bottom_roi_batch_inds, 
    const Dtype* top_data_buffer, 
    const Dtype* top_diff,
    Dtype* bottom_diff_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) is an element in the input 
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int roi_batch_ind = bottom_roi_batch_inds[nth_roi] - 1;
 
    if (roi_batch_ind == n) {
      int bin_size_ind = nth_roi * 2;
      Dtype bin_size_w = max(bottom_bin_sizes[bin_size_ind+0], MIN_BIN_SIZE);
      Dtype bin_size_h = max(bottom_bin_sizes[bin_size_ind+1], MIN_BIN_SIZE);

      ///** for debug **/ 
      //int top_buffer_index = nth_roi * (pooled_height * pooled_width * NUM_BUFFERS) + h * (pooled_width * NUM_BUFFERS) + w * NUM_BUFFERS;
      ////gain_x_all = top_data_buffer[top_buffer_index+0];
      ////gain_y_all = top_data_buffer[top_buffer_index+1];
      ////bottom_diff_data[index] = top_data_buffer[top_buffer_index+1];
      ///** til here **//

      int grid_ctr_ind = nth_roi * (pooled_height * pooled_width * 2) +  0 * (pooled_width * 2) + 0 * 2;
      //Dtype roi_start_w = bottom_grid_ctrs[grid_ctr_ind+0] - bin_size_w / 2.0;
      //Dtype roi_start_h = bottom_grid_ctrs[grid_ctr_ind+1] - bin_size_w / 2.0;

      // wstart = floor(roi_start_w + pw * bin_size_w + bin_size_w / 2)
      // -> wstart - roi_start_w - bin_size_w / 2 = bin_size_w * pw
      // hstart = floor(roi_start_h + ph * bin_size_h + bin_size_h / 2)
      // -> hstart - roi_start_h - bin_size_h / 2 = bin_size_h * pw

      int pwstart = pooled_width, pwend = -1;
      int wstart = 0, wend = 0;
      Dtype wctr, wstart_, wend_;
      for (int pw = 0; pw < pooled_width; pw++) {
        //wctr = roi_start_w + pw * bin_size_w + bin_size_w / 2.0;
        /*int*/ grid_ctr_ind = nth_roi * (pooled_height * pooled_width * 2) +  0 * (pooled_width * 2) + pw * 2;
        wctr = bottom_grid_ctrs[grid_ctr_ind+0] - 1;
        wstart_ = wctr - bin_size_w / 2.0;
        wend_   = wctr + bin_size_w / 2.0;

        wstart = static_cast<int>(floor(wstart_));
        wend   = static_cast<int>( ceil(wend_));

        //wstart = min(max(wstart, 0), width -1);
        //wend   = min(max(wend, 0),   width -1);
     
        if ((wstart <= w) && (w <= wend)) {
          if (pw < pwstart) {
            pwstart = pw;
          }
          if (pw > pwend) {
            pwend   = pw;
          } 
        }
      }

      int phstart = pooled_height, phend = -1;
      int hstart = 0, hend = 0;
      Dtype hctr, hstart_, hend_; 
      for (int ph = 0; ph < pooled_height; ++ph) {
        /*int*/ grid_ctr_ind = nth_roi * (pooled_height * pooled_width * 2) +  ph * (pooled_width * 2) + 0 * 2;

        hctr = bottom_grid_ctrs[grid_ctr_ind+1] - 1;
        hstart_ = hctr - bin_size_h / 2.0;
        hend_   = hctr + bin_size_h / 2.0;

        hstart = static_cast<int>(floor(hstart_));
        hend   = static_cast<int>( ceil(hend_));

        //hstart = min(max(hstart, 0), height-1);
        //hend   = min(max(hend, 0),   height-1);

        if (hstart <= h && h <= hend) {
          if (ph < phstart) {
            phstart = ph;
          }
          if (ph > phend) {
            phend   = ph;
          }
        }
      }
 
      //bottom_diff_data[index] = bottom_grid_ctrs[grid_ctr_ind+0] - 1;  
      //bottom_diff_data[index] = bottom_grid_ctrs[nth_roi * (pooled_height * pooled_width * 2) +  0 * (pooled_width * 2) + 2 * 2 + 0] - bin_size_w / 2.0;
      //bottom_diff_data[index] = (static_cast<dtype>(w) - roi_start_w - bin_size_w / 2.0) * bin_size_pw + 1; 
      //bottom_diff_data[index] = (static_cast<dtype>(w+1) - roi_start_w - bin_size_w / 2.0) * bin_size_pw + 1; 
      //bottom_diff_data[index] = phend+1; //pwend+1;
      //bottom_diff_data[index] = phstart+1; //pwstart+1;
      //bottom_diff_data[index] = roi_start_w + 1; 
      //bottom_diff_data[index] = roi_start_h + 1; 
 
      // Clip to top boundaries
      phstart = min(max(phstart, 0), pooled_height-1);
      phend =   min(max(phend, 0),   pooled_height-1);
      pwstart = min(max(pwstart, 0), pooled_width -1);
      pwend =   min(max(pwend, 0),   pooled_width -1);
  
      Dtype w_ = w, h_ = h; 
      //Dtype wctr = 0, hctr = 0;
      Dtype gain = 0, gain_x = 0, gain_y = 0, gain_x_all = 0, gain_y_all = 0;  
      for (int ph = phstart; ph <= phend; ++ph) {
        for (int pw = pwstart; pw <= pwend; ++pw) {
          int top_index = nth_roi * (channels * pooled_height * pooled_width) + c * (pooled_height * pooled_width) + ph * pooled_width  + pw;
          int top_buffer_index = nth_roi * (pooled_height * pooled_width * NUM_BUFFERS) + ph * (pooled_width * NUM_BUFFERS) + pw * NUM_BUFFERS;
          /*int*/ grid_ctr_ind = nth_roi * (pooled_height * pooled_width * 2) + ph * (pooled_width * 2) + pw * 2;

          wctr = bottom_grid_ctrs[grid_ctr_ind+0] - 1;
          hctr = bottom_grid_ctrs[grid_ctr_ind+1] - 1;
          gain_x_all = top_data_buffer[top_buffer_index+0]; 
          gain_y_all = top_data_buffer[top_buffer_index+1]; 
  
          wstart_ = wctr - bin_size_w / 2.0;
          wend_   = wctr + bin_size_w / 2.0;
          hstart_ = hctr - bin_size_h / 2.0;
          hend_   = hctr + bin_size_h / 2.0;

          //gain_x = (bin_size_w+1) - abs(w_ - wctr); 
          //gain_y = (bin_size_h+1) - abs(h_ - hctr);
          gain_x = ((wstart_ <= w_ && w_ <= wend_) ? 1.0 : 0) * (bin_size_w - abs(w_ - wctr));
          gain_y = ((hstart_ <= h_ && h_ <= hend_) ? 1.0 : 0) * (bin_size_h - abs(h_ - hctr));

          if (gain_x_all > PRECISION_LIMIT)
            gain_x = gain_x / gain_x_all; 
          if (gain_y_all > PRECISION_LIMIT)  
            gain_y = gain_y / gain_y_all; 
  
          gain = gain_x * gain_y;
          bottom_diff_data[index] = bottom_diff_data[index] + gain * top_diff[top_index];
        }
      }

    }
  }
}

template <typename Dtype>
__global__ void ROIWarpBilinearBackwardGridCtrs(
    const int nthreads, 
    const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
    //const int c,  
    const Dtype* bottom_data, 
    const Dtype* bottom_grid_ctrs, 
    const Dtype* bottom_bin_sizes, 
    const Dtype* bottom_roi_batch_inds, 
    const Dtype* top_data_buffer, 
    const Dtype* top_diff,
    Dtype* bottom_diff_grid_ctrs_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw, d) is an element in the grid_ctrs_buffer 
    int d  = index % 2; 
    int pw = (index / 2) % pooled_width;
    int ph = (index / 2 / pooled_width) % pooled_height;
    int c  = (index / 2 / pooled_width / pooled_height) % channels;
    int n  =  index / 2 / pooled_width / pooled_height / channels;

    // get top buffer index and top buffers
    int top_buffer_index = n * (pooled_height * pooled_width * NUM_BUFFERS) + ph * (pooled_width * NUM_BUFFERS) + pw * NUM_BUFFERS;
    Dtype gain_x_all          = top_data_buffer[top_buffer_index+0];
    Dtype gain_y_all          = top_data_buffer[top_buffer_index+1];
    Dtype dgx_final_dwctr_all = top_data_buffer[top_buffer_index+2];
    Dtype dgy_final_dhctr_all = top_data_buffer[top_buffer_index+3];

    // get top index 
    int top_index = n * (channels * pooled_height * pooled_width) + c * (pooled_height * pooled_width) + ph * pooled_width + pw; 

    // estimate grad
    int roi_batch_ind = bottom_roi_batch_inds[n] - 1;
    int grid_ctr_ind = n * (pooled_height * pooled_width * 2) +  ph * (pooled_width * 2) + pw * 2;
    int bin_size_ind = n * 2;

    Dtype wctr = bottom_grid_ctrs[grid_ctr_ind+0] - 1;
    Dtype hctr = bottom_grid_ctrs[grid_ctr_ind+1] - 1;

    Dtype bin_size_w = max(bottom_bin_sizes[bin_size_ind+0], MIN_BIN_SIZE);
    Dtype bin_size_h = max(bottom_bin_sizes[bin_size_ind+1], MIN_BIN_SIZE);
 
    Dtype wstart_ = wctr - bin_size_w / 2.0;
    Dtype hstart_ = hctr - bin_size_h / 2.0;
    Dtype wend_   = wctr + bin_size_w / 2.0;
    Dtype hend_   = hctr + bin_size_h / 2.0;
              
    int wstart = static_cast<int>(floor(wstart_)); 
    int hstart = static_cast<int>(floor(hstart_)); 
    int wend   = static_cast<int>( ceil(wend_));// + 1;
    int hend   = static_cast<int>( ceil(hend_));// + 1;
  
    //// Add roi offsets and clip to input boundaries
    //hstart = min(max(hstart, 0), height);
    //hend   = min(max(hend, 0),   height);
    //wstart = min(max(wstart, 0), width );
    //wend   = min(max(wend, 0),   width );

    // Auxilliary variables used in backprop 
    Dtype w_mask = 0, h_mask = 0; 
    //Dtype dgx_final_dwctr_all  = 0;
    //Dtype dgy_final_dhctr_all  = 0;

    // output = g * input
    // do / dwctr = input * dg / dwctr
    // g = gx_final * gy_final 
    // gx_final = gx / gx_all 
    // dg / dwctr = dg / dgx_final * dgx_final / dwctr 
    //            =  gy_final      * ( dgx/dwctr  * gx_all - gx * dgx_all/dwctr  ) / (gx_all)^2
    //            =  gy_final      * ( (w >= wctr ? 1 : -1) * gx_all - gx * sum_for_w{ (w >= wctr ? 1 : -1) } ) / gx_all^2
 
    // Define an empty pooling region to be zero
    Dtype val = 0;  
    Dtype gain = 0, gain_x = 0, gain_y = 0;
    Dtype coeff_x = 0, coeff_y = 0;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h <= hend; ++h) {
      Dtype h_ = h;
      h_mask = ((hstart_ <= h_ && h_ <= hend_) ? 1.0 : 0); 
      for (int w = wstart; w <= wend; ++w) {
        int bottom_index = h * width + w;
        Dtype w_ = w;
        w_mask = ((wstart_ <= w_ && w_ <= wend_) ? 1.0 : 0);

        if (0 <= h && h < height && 0 <= w && w < width) {
          //gain_x = (bin_size_w+1) - abs(w_ - wctr);
          //gain_y = (bin_size_h+1) - abs(h_ - hctr);
          gain_x = w_mask * (bin_size_w - abs(w_ - wctr));
          gain_y = h_mask * (bin_size_h - abs(h_ - hctr));
 
          if (d == 0) {
            coeff_x = gain_y * bottom_data[bottom_index];
            if (gain_x_all > PRECISION_LIMIT) {coeff_x = coeff_x / (gain_x_all*gain_x_all);}
            if (gain_y_all > PRECISION_LIMIT) {coeff_x = coeff_x / gain_y_all;}
            val = val + ((w_ >= wctr ? 1 : -1) * gain_x_all - gain_x * dgx_final_dwctr_all ) * coeff_x;
          }
          else if (d == 1) {
            coeff_y = gain_x * bottom_data[bottom_index];
            if (gain_y_all > PRECISION_LIMIT) {coeff_y = coeff_y / (gain_y_all*gain_y_all);}
            if (gain_x_all > PRECISION_LIMIT) {coeff_y = coeff_y / gain_x_all;}
            val = val + ((h >= hctr ? 1 : -1) * gain_y_all - gain_y * dgy_final_dhctr_all ) * coeff_y;
          }
          /** for debug **/ 
          //gain = gain_x * gain_y; 
          //if (gain_x_all > PRECISION_LIMIT) { gain = gain / gain_x_all; } 
          //if (gain_y_all > PRECISION_LIMIT) { gain = gain / gain_y_all; }
          //val = val + gain * bottom_data[bottom_index]; 
          ////val = val + gain; 
          /** til here **/
        }
      }
    }
    bottom_diff_grid_ctrs_buffer[index] = val * top_diff[top_index];

    /** for debug **/ 
    //bottom_diff_grid_ctrs_buffer[index] = top_diff[top_index];
    //bottom_diff_grid_ctrs_buffer[index] = dgx_final_dwctr_all;
    //bottom_diff_grid_ctrs_buffer[index] = dgy_final_dhctr_all;
    //bottom_diff_grid_ctrs_buffer[index] = gain_x_all;
    //bottom_diff_grid_ctrs_buffer[index] = gain_y_all;
    //bottom_diff_grid_ctrs_buffer[index] = val;
    //bottom_diff_grid_ctrs_buffer[index] = d+1;
    //bottom_diff_grid_ctrs_buffer[index] = pw+1;
    //bottom_diff_grid_ctrs_buffer[index] = ph+1;
    /** til here **/ 
  }
}

template <typename Dtype>
__global__ void ROIWarpBilinearBackwardBinSizes(
    const int nthreads, 
    const int channels, const int height, const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_data, 
    const Dtype* bottom_grid_ctrs, 
    const Dtype* bottom_bin_sizes, 
    const Dtype* bottom_roi_batch_inds, 
    const Dtype* top_data_buffer, 
    const Dtype* top_diff,
    Dtype* bottom_diff_bin_sizes_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw, d) is an element in the grid_ctrs_buffer 
    int d  = index % 2; 
    int pw = (index / 2) % pooled_width;
    int ph = (index / 2 / pooled_width) % pooled_height;
    int c  = (index / 2 / pooled_width / pooled_height) % channels;
    int n  =  index / 2 / pooled_width / pooled_height / channels;

    // get top buffer index and top buffers
    int top_buffer_index = n * (pooled_height * pooled_width * NUM_BUFFERS) + ph * (pooled_width * NUM_BUFFERS) + pw * NUM_BUFFERS;
    Dtype gain_x_all           = top_data_buffer[top_buffer_index+0];
    Dtype gain_y_all           = top_data_buffer[top_buffer_index+1];
    Dtype dgx_final_dwdiff_all = top_data_buffer[top_buffer_index+4];
    Dtype dgy_final_dhdiff_all = top_data_buffer[top_buffer_index+5];

    // get top index 
    int top_index = n * (channels * pooled_height * pooled_width) + c * (pooled_height * pooled_width) + ph * pooled_width + pw; 

    // estimate grad
    int roi_batch_ind = bottom_roi_batch_inds[n] - 1;
    int grid_ctr_ind = n * (pooled_height * pooled_width * 2) +  ph * (pooled_width * 2) + pw * 2;
    int bin_size_ind = n * 2;

    Dtype wctr = bottom_grid_ctrs[grid_ctr_ind+0] - 1;
    Dtype hctr = bottom_grid_ctrs[grid_ctr_ind+1] - 1;

    Dtype bin_size_w = bottom_bin_sizes[bin_size_ind+0];
    Dtype bin_size_h = bottom_bin_sizes[bin_size_ind+1];
 
    Dtype wstart_ = wctr - bin_size_w / 2.0;
    Dtype hstart_ = hctr - bin_size_h / 2.0;
    Dtype wend_   = wctr + bin_size_w / 2.0;
    Dtype hend_   = hctr + bin_size_h / 2.0;
              
    int wstart = static_cast<int>(floor(wstart_)); 
    int hstart = static_cast<int>(floor(hstart_)); 
    int wend   = static_cast<int>( ceil(wend_)); 
    int hend   = static_cast<int>( ceil(hend_));
  
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height-1);
    hend   = min(max(hend, 0),   height-1);
    wstart = min(max(wstart, 0), width -1);
    wend   = min(max(wend, 0),   width -1);

    // Define an empty pooling region to be zero
    Dtype val = 0;  
    Dtype gain = 0, gain_x = 0, gain_y = 0;
    Dtype coeff_x = 0, coeff_y = 0;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h <= hend; ++h) {
      Dtype h_ = h;
      for (int w = wstart; w <= wend; ++w) {
        int bottom_index = h * width + w;
        Dtype w_ = w;

        gain_x = (bin_size_w+1) - abs(w_ - wctr);
        gain_y = (bin_size_h+1) - abs(h_ - hctr);

        if (d == 0) {
          coeff_x = gain_y * bottom_data[bottom_index];
          if (gain_x_all > PRECISION_LIMIT) {coeff_x = coeff_x / (gain_x_all*gain_x_all);}
          if (gain_y_all > PRECISION_LIMIT) {coeff_x = coeff_x / gain_y_all;}
          val = val + (1 * gain_x_all - gain_x * dgx_final_dwdiff_all) * coeff_x;
        }
        else if (d == 1) {
          coeff_y = gain_x * bottom_data[bottom_index];
          if (gain_y_all > PRECISION_LIMIT) {coeff_y = coeff_y / (gain_y_all*gain_y_all);}
          if (gain_x_all > PRECISION_LIMIT) {coeff_y = coeff_y / gain_x_all;}
          val = val + (1 * gain_y_all - gain_y * dgy_final_dhdiff_all) * coeff_y;
        }
        /** for debug **/ 
        //gain = gain_x * gain_y; 
        //if (gain_x_all > PRECISION_LIMIT) { gain = gain / gain_x_all; } 
        //if (gain_y_all > PRECISION_LIMIT) { gain = gain / gain_y_all; }
        //val = val + gain * bottom_data[bottom_index]; 
        ////val = val + gain; 
        /** til here **/ 
      }
    }
    bottom_diff_bin_sizes_buffer[index] = val * top_diff[top_index];

    /** for debug **/ 
    //bottom_diff_grid_ctrs_buffer[index] = top_diff[top_index];
    //bottom_diff_grid_ctrs_buffer[index] = dgx_final_dwctr_all;
    //bottom_diff_grid_ctrs_buffer[index] = dgy_final_dhctr_all;
    //bottom_diff_grid_ctrs_buffer[index] = gain_x_all;
    //bottom_diff_grid_ctrs_buffer[index] = val;
    //bottom_diff_grid_ctrs_buffer[index] = d+1;
    //bottom_diff_grid_ctrs_buffer[index] = pw+1;
    //bottom_diff_grid_ctrs_buffer[index] = ph+1;
    /** til here **/ 
  }
}

extern "C"
void inn_ROIWarpingBilinearSample_updateGradInput(THCState *state,
    THCudaTensor *gradInput_data,      THCudaTensor *data,
    THCudaTensor *gradInput_grid_ctrs, THCudaTensor *grid_ctrs, THCudaTensor *gradInput_grid_ctrs_buffer, 
    THCudaTensor *gradInput_bin_sizes, THCudaTensor *bin_sizes, THCudaTensor *gradInput_bin_sizes_buffer,
    THCudaTensor *roi_batch_inds,
    THCudaTensor *output_buffer,
    THCudaTensor *gradOutput,
    int pooled_height, int pooled_width)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, grid_ctrs) == 4 && grid_ctrs->size[3] == 2);
  THAssert(THCudaTensor_nDimension(state, bin_sizes) == 2 && bin_sizes->size[1] == 2);
  THAssert(THCudaTensor_nDimension(state, roi_batch_inds) == 2
           && roi_batch_inds->size[0] == grid_ctrs->size[0]
           && roi_batch_inds->size[0] == bin_sizes->size[0]);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, grid_ctrs));
  THAssert(THCudaTensor_isContiguous(state, bin_sizes));

  long num_rois = grid_ctrs->size[0];
  long nInputPlane = data->size[1];

  long count = 0; 

  // backpropagation for data
  for (int nth_roi = 0; nth_roi < num_rois; ++nth_roi) {
    count = THCudaTensor_nElement(state, gradInput_data);
    ROIWarpBilinearBackwardData<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
        count,
        nInputPlane, data->size[2], data->size[3], pooled_height, pooled_width,
        nth_roi,
        THCudaTensor_data(state, grid_ctrs),
        THCudaTensor_data(state, bin_sizes),
        THCudaTensor_data(state, roi_batch_inds),
        THCudaTensor_data(state, output_buffer),
        THCudaTensor_data(state, gradOutput),
        THCudaTensor_data(state, gradInput_data)
        );
  }

  // backpropagation for grid_ctrs 
  count = THCudaTensor_nElement(state, gradInput_grid_ctrs_buffer);
  ROIWarpBilinearBackwardGridCtrs<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      nInputPlane, data->size[2], data->size[3], pooled_height, pooled_width,
      THCudaTensor_data(state, data),
      THCudaTensor_data(state, grid_ctrs),
      THCudaTensor_data(state, bin_sizes),
      THCudaTensor_data(state, roi_batch_inds),
      THCudaTensor_data(state, output_buffer),
      THCudaTensor_data(state, gradOutput),
      THCudaTensor_data(state, gradInput_grid_ctrs_buffer)
      );

  // backpropagation for bin_sizes 
  count = THCudaTensor_nElement(state, gradInput_bin_sizes_buffer);
  ROIWarpBilinearBackwardBinSizes<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      nInputPlane, data->size[2], data->size[3], pooled_height, pooled_width,
      THCudaTensor_data(state, data),
      THCudaTensor_data(state, grid_ctrs),
      THCudaTensor_data(state, bin_sizes),
      THCudaTensor_data(state, roi_batch_inds),
      THCudaTensor_data(state, output_buffer),
      THCudaTensor_data(state, gradOutput),
      THCudaTensor_data(state, gradInput_bin_sizes_buffer)
      );

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inn_ROIWarpingBilinearSample_updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

#undef NUM_BUFFERS 
#undef PRECISION_LIMIT
#undef MIN_BIN_SIZE 
