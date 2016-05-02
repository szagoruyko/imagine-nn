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


using std::max;
using std::min;


template <typename Dtype>
__global__ void ROIWarpForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, const Dtype* bottom_delta_rois, Dtype* top_data, Dtype* top_data_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = (bottom_rois[0] - 1);
    //int roi_start_w = round((bottom_rois[1] - 1) * spatial_scale);
    //int roi_start_h = round((bottom_rois[2] - 1)* spatial_scale);
    //int roi_end_w = round((bottom_rois[3] - 1) * spatial_scale);
    //int roi_end_h = round((bottom_rois[4] - 1) * spatial_scale);

    Dtype src_w = bottom_rois[3] - bottom_rois[1] + 1; 
    Dtype src_h = bottom_rois[4] - bottom_rois[2] + 1;
    Dtype src_ctr_x = bottom_rois[1] + 0.5*(src_w-1.0); 
    Dtype src_ctr_y = bottom_rois[2] + 0.5*(src_h-1.0); 

    Dtype dst_ctr_x = bottom_delta_rois[1]; // dx (in fast-rcnn notation) = cx (in here)
    Dtype dst_ctr_y = bottom_delta_rois[2]; // dy (in fast-rcnn notation) = cy (in here) 
    Dtype dst_scl_x = bottom_delta_rois[3]; // dw (in fast-rcnn notation) = sx (in here)
    Dtype dst_scl_y = bottom_delta_rois[4]; // dh (in fast-rcnn notation) = sy (in here) 

    Dtype pred_ctr_x = dst_ctr_x * src_w + src_ctr_x; // dpcx / dcx = src_w
    Dtype pred_ctr_y = dst_ctr_y * src_h + src_ctr_y; // dpcy / dcy = src_h
    Dtype pred_w = exp(dst_scl_x) * src_w;            // dpw  / dsx = src_w * exp(dsx)  
    Dtype pred_h = exp(dst_scl_y) * src_h;            // dph  / dsy = src_h * exp(dsy)  
    
    Dtype roi_start_w = ( (pred_ctr_x - 0.5*(pred_w-1)) - 1 ) * spatial_scale; // drsw / dpcx = spatial_scale; drsw / dpw = -0.5 * spatial_scale
    Dtype roi_start_h = ( (pred_ctr_y - 0.5*(pred_h-1)) - 1 ) * spatial_scale; // drsh / dpcy = spatial_scale; drsh / dph = -0.5 * spatial_scale
    Dtype roi_end_w =   ( (pred_ctr_x + 0.5*(pred_w-1)) - 1 ) * spatial_scale; // drew / dpcx = spatial_scale; drew / dpw =  0.5 * spatial_scale
    Dtype roi_end_h =   ( (pred_ctr_y + 0.5*(pred_h-1)) - 1 ) * spatial_scale; // dreh / dpcy = spatial_scale; dreh / dph =  0.5 * spatial_scale
    assert(roi_end_w - roi_start_w >= 0);
    assert(roi_end_h - roi_start_h >= 0);   
    
    // drsw / dcx = drsw / dpcx * dpcx / dcx = spatial_scale * src_w
    // drew / dcx = drew / dpcx * dpcx / dcx = spatial_scale * src_w

    // drsh / dcy = drsh / dpcy * dpcy / dcy = spatial_scale * src_h
    // dreh / dcy = dreh / dpcy * dpcy / dcy = spatial_scale * src_h

    // drsw / dsx = drsw / dpw * dpw / dsx = -0.5 * spatial_scale * src_w * exp(dsx) 
    // drew / dsx = drew / dpw * dpw / dsx =  0.5 * spatial_scale * src_w * exp(dsx)
 
    // drsh / dsy = drsh / dph * dph / dsy = -0.5 * spatial_scale * src_h * exp(dsy)
    // dreh / dsy = dreh / dph * dph / dsy =  0.5 * spatial_scale * src_h * exp(dsy) 
 
    // Force malformed ROIs to be 1x1
    Dtype roi_width  = roi_end_w - roi_start_w + 1; // drw / drew =  1
                                                    // drw / drsw = -1
    Dtype roi_height = roi_end_h - roi_start_h + 1; // drh / dreh =  1
                                                    // drh / drsh = -1
    // drw / dcx = drw / drew * drew / dcx + drw / drsw * drsw / dcx = drew / dcx - drsw / dcx = spatial_scale * src_w - spatial_scale * src_w = 0 
    // drh / dcy = drh / dreh * dreh / dcy + drh / drsh * drsh / dcy = dreh / dcy - drsh / dcy = spatial_scale * src_h - spatial_scale * src_h = 0 
    // drw / dsx = drw / drew * drew / dsx + drw / drsw * drsw / dsx = drew / dsx - drsw / dsx = 0.5 * spatial_scale * src_w * exp(dsx) - (-0.5 * spatial_scale * src_w * exp(dsx)) = spatial_scale * src_w * exp(dsx) 
    // drh / dsy = drh / dreh * dreh / dsy + drh / drsh * drsh / dsy = dreh / dsy - drsh / dsy = 0.5 * spatial_scale * src_h * exp(dsy) - (-0.5 * spatial_scale * src_h * exp(dsy)) = spatial_scale * src_h * exp(dsy) 
    
    Dtype bin_size_w = roi_width  / static_cast<Dtype>(pooled_width);  // dbw / drw  =  1 / pooled_width
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height); // dbh / drh  =  1 / pooled_height
    // dbw / dcx = dbw / drw * drw / dcx = 0 
    // dbh / dcy = dbh / drh * drh / dcy = 0
    // dbw / dsx = dbw / drw * drw / dsx = 1 / pooled_width * spatial_scale * src_w * exp(dsx) 
    // dbh / dsy = dbh / drh * drh / dsy = 1 / pooled_height * spatial_scale * src_h * exp(dsy) 

    //int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)        // dws / dbw = pw 
    //                                    * bin_size_w)) + roi_start_w; // dws / drsw = 1
    //int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)        // dhs / dbh = ph 
    //                                    * bin_size_h)) + roi_start_h; // dhs / drsh = 1 
    //int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)       // dwe / dbw = (pw+1)
    //                                 * bin_size_w)) + roi_start_w;    // dwe / drsw = 1 
    //int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)       // dhe / dbh = (ph+1)
    //                                 * bin_size_h)) + roi_start_h;    // dhe / drsh = 1
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)        // dws / dbw = pw
                                        * bin_size_w + roi_start_w)); // dws / drsw = 1
    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)        // dhs / dbh = ph
                                        * bin_size_h + roi_start_h)); // dhs / drsh = 1
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)       // dwe / dbw = (pw+1)
                                     * bin_size_w + roi_start_w));    // dwe / drsw = 1
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)       // dhe / dbh = (ph+1)
                                     * bin_size_h + roi_start_h));    // dhe / drsh = 1 
    // dws / dcx = dws / dbw * dbw / dcx + dws / drsw * drsw / dcx = pw * 0 + 1 * spatial_scale * src_w     = spatial_scale * src_w
    // dwe / dcx = dwe / dbw * dbw / dcx + dwe / drsw * drsw / dcx = (pw+1) * 0 + 1 * spatial_scale * src_w = spatial_scale * src_w

    // dws / dsx = dws / dbw * dbw / dsx + dws / drsw * drsw / dsx = pw * 1 / pooled_width * spatial_scale * src_w * exp(dsx) + 1 * (-0.5) * spatial_scale * src_w * exp(dsx) = ( pw / pooled_width - 0.5 ) * spatial_scale * src_w * exp(dsx) 
    // dwe / dsx = dwe / dbw * dbw / dsx + dwe / drsw * drsw / dsx = (pw+1) * 1 / pooled_width * spatial_scale * src_w * exp(dsx) + 1 * 0.5 * spatial_scale * src_w * exp(dsx) = ( (pw+1)/pooled_width + 0.5 ) * spatial_scale * src_w * exp(dsx)

    // dhs / dcy = dhs / dbh * dbh / dcy + dhs / drsh * drsh / dcy = ph * 0 + 1 * spatial_scale * src_h     = spatial_scale * src_w
    // dhe / dcy = dhe / dbh * dbh / dcy + dhe / drsh * drsh / dcy = (ph+1) * 0 + 1 * spatial_scale * src_h = spatial_scale * src_h

    // dhs / dsy = dhs / dbh * dbh / dsy + dhs / drsh * drsh / dsy = ph * 1 / pooled_height * spatial_scale * src_h * exp(dsy) + 1 * (-0.5) * spatial_scale * src_h * exp(dsy) = (ph / pooled_height - 0.5) * spatial_scale * src_h * exp(dsy) 
    // dhe / dsy = dhe / dbh * dbh / dsy + dhe / drsh * drsh / dsy = (ph+1) * 1 / pooled_height * spatial_scale * src_h * exp(dsy) + 1 * 0.5 * spatial_scale * src_h * exp(dsy) = ((ph+1)/pooled_height + 0.5) * spatial_scale * src_h * exp(dsy)  

    //top_data[index] = static_cast<Dtype>(hend-1-hstart)+1;
    //top_data[index] = hend;
    //top_data[index] = wend; //hend;
    //top_data[index] = wstart+1; //hstart+1;

    Dtype wctr = static_cast<Dtype>(wend-1+wstart) * 0.5;    // dwctr / dwe = 0.5; dwctr / dws = 0.5 
    Dtype hctr = static_cast<Dtype>(hend-1+hstart) * 0.5;    // dhctr / dhe = 0.5; dhctr / dhs = 0.5 
    Dtype wdiff = max(static_cast<Dtype>(wend-1-wstart), 1.);         // dwdiff / dwe = 1; dwdiff / dws = -1
    Dtype hdiff = max(static_cast<Dtype>(hend-1-hstart), 1.);         // dhdiff / dhe = 1; dhdiff / dhs = -1
    // dwctr / dcx = dwctr / dwe * dwe / dcx + dwctr / dws * dws / dcx = 0.5 * spatial_scale * src_w + 0.5 * spatial_scale * src_w = spatial_scale * src_w 
    // dwdiff / dcx = dwdiff / dwe * dwe / dcx + dwdiff / dws * dws / dcx = 1 * spatial_scale * src_w -  1  * spatial_scale * src_w = 0 

    // dhctr / dcy = spatial_scale * src_h
    // dhdiff / dcy = 0
  
    // dwctr / dsx = dwctr / dwe * dwe / dsx + dwctr / dws * dws / dsx = 0.5 * ((pw+1)/pooled_width + 0.5) * spatial_scale * src_w * exp(dsx) + 0.5 * (pw/pooled_width - 0.5) * spatial_scale * src_w * exp(dsx)
    //                                                                 = 0.5 * (2*pw+1)/pooled_width * spatial_scale * src_w * exp(dsx)
    //                                                                 = (pw + 0.5) / pooled_width * spatial_scale * src_w * exp(dsx)
    // dwdiff / dsx = dwdiff / dwe * dwe / dsx + dwdiff / dws * dws / dsx = 1 * ((pw+1)/pooled_width + 0.5) * spatial_scale * src_w * exp(dsx) + (-1) * (pw/pooled_width - 0.5) * spatial_scale * src_w * exp(dsx)
    //                                                                    = (wend-wstart) >= 1 ? (1 / pooled_width + 1) * spatial_scale * src_w * exp(dsx) : 0
    // dhctr / dsy  = (ph + 0.5) / pooled_height * spatial_scale * src_h * exp(dsy)
    // dhdiff / dsy = (hend-hstart) >= 1 ? (1 / pooled_height + 1) * spatial_scale * src_h * exp(dsy) : 0
    
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);         //  
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);

    // Define an empty pooling region to be zero
    Dtype val = 0; Dtype gain = 0, gain_x = 0, gain_y = 0, gain_x_all = 0, gain_y_all = 0;   
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        Dtype w_ = w, h_ = h;  
        //gain_x = max(0., 1 - abs( dst_ctr_x + static_cast<Dtype>(pw) / static_cast<Dtype>(pooled_width) * dst_scl_x - w ));
        //gain_y = max(0., 1 - abs( dst_ctr_y + static_cast<Dtype>(ph) / static_cast<Dtype>(pooled_height) * dst_scl_y - h));
        gain_x = (wdiff - abs((w_ - wctr))) / wdiff; 
        gain_y = (hdiff - abs((h_ - hctr))) / hdiff;   
        gain = gain_x * gain_y;

        val = val + gain * bottom_data[bottom_index];
        //val = val + gain;
        if (h == hstart) 
          gain_x_all = gain_x_all + gain_x;
      }
      gain_y_all = gain_y_all + gain_y;
    }
    if (gain_x_all > 1e-10)
      val = val / gain_x_all;
    if (gain_y_all > 1e-10)  
      val = val / gain_y_all;
    top_data[index] = val;

    //top_data[index] = gain_x_all; 
    //top_data[index] = gain_y_all; 
    int buffer_index = n * (channels * pooled_height * pooled_width * 6) + c * (pooled_height * pooled_width * 6) + ph * (pooled_width * 6) + pw * 6;
    top_data_buffer[buffer_index+0] = wctr;
    top_data_buffer[buffer_index+1] = wdiff;
    top_data_buffer[buffer_index+2] = hctr;
    top_data_buffer[buffer_index+3] = hdiff; 
    top_data_buffer[buffer_index+4] = gain_x_all; 
    top_data_buffer[buffer_index+5] = gain_y_all; 
  }
}

extern "C"
void inn_ROIWarping_updateOutput(THCState *state,
    THCudaTensor *output, THCudaTensor *output_buffer,
    THCudaTensor *data, THCudaTensor* rois, THCudaTensor* delta_rois, int W, int H, double spatial_scale)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, rois) == 2 && rois->size[1] == 5);
  THAssert(THCudaTensor_nDimension(state, delta_rois) == 2 && delta_rois->size[1] == 5);
  THAssert(THCudaTensor_nDimension(state, rois) == THCudaTensor_nDimension(state, delta_rois) &&
           rois->size[0] == delta_rois->size[0] &&
           rois->size[1] == delta_rois->size[1]);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, rois));
  THAssert(THCudaTensor_isContiguous(state, delta_rois));
  long num_rois = rois->size[0];
  long nInputPlane = data->size[1];
  THCudaTensor_resize4d(state, output, num_rois, nInputPlane, H, W);
  THCudaTensor_resize5d(state, output_buffer, num_rois, nInputPlane, H, W, 6);
  //THCudaTensor_zero(state, output_buffer);

  long count = THCudaTensor_nElement(state, output);

  ROIWarpForward<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      count,
      THCudaTensor_data(state, data),
      spatial_scale, nInputPlane, data->size[2], data->size[3], H, W,
      THCudaTensor_data(state, rois),
      THCudaTensor_data(state, delta_rois),
      THCudaTensor_data(state, output),
      THCudaTensor_data(state, output_buffer)
      );

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inn_ROIWarping_updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}

template <typename Dtype>
__global__ void ROIWarpBackwardData(const int nthreads, const Dtype* top_data_buffer,
    const Dtype spatial_scale, const int channels, const int height, const int width, 
    const int pooled_height, const int pooled_width, const int nth_roi, 
    const Dtype* bottom_rois, const Dtype* bottom_delta_rois, 
    const Dtype* top_diff,
    Dtype* bottom_diff_data, 
    Dtype* bottom_diff_delta_rois_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    // (n, c, h, w) is an element in the pooled output
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    bottom_rois += nth_roi * 5;
    int roi_batch_ind = (bottom_rois[0] - 1);
 
    if (roi_batch_ind == n) {

      Dtype src_w = bottom_rois[3] - bottom_rois[1] + 1;
      Dtype src_h = bottom_rois[4] - bottom_rois[2] + 1;
      Dtype src_ctr_x = bottom_rois[1] + 0.5*(src_w-1.0);
      Dtype src_ctr_y = bottom_rois[2] + 0.5*(src_h-1.0);
  
      Dtype dst_ctr_x = bottom_delta_rois[1]; // dx (in fast-rcnn notation) = cx (in here)
      Dtype dst_ctr_y = bottom_delta_rois[2]; // dy (in fast-rcnn notation) = cy (in here)
      Dtype dst_scl_x = bottom_delta_rois[3]; // dw (in fast-rcnn notation) = sx (in here)
      Dtype dst_scl_y = bottom_delta_rois[4]; // dh (in fast-rcnn notation) = sy (in here)
  
      Dtype pred_ctr_x = dst_ctr_x * src_w + src_ctr_x; // dpcx / dcx = src_w
      Dtype pred_ctr_y = dst_ctr_y * src_h + src_ctr_y; // dpcy / dcy = src_h
      Dtype pred_w = exp(dst_scl_x) * src_w;            // dpw  / dsx = src_w * exp(dsx)
      Dtype pred_h = exp(dst_scl_y) * src_h;            // dph  / dsy = src_h * exp(dsy)
  
      Dtype roi_start_w = ( (pred_ctr_x - 0.5*(pred_w-1)) - 1 ) * spatial_scale; // drsw / dpcx = spatial_scale; drsw / dpw = -0.5 * spatial_scale
      Dtype roi_start_h = ( (pred_ctr_y - 0.5*(pred_h-1)) - 1 ) * spatial_scale; // drsh / dpcy = spatial_scale; drsh / dph = -0.5 * spatial_scale
      Dtype roi_end_w =   ( (pred_ctr_x + 0.5*(pred_w-1)) - 1 ) * spatial_scale; // drew / dpcx = spatial_scale; drew / dpw =  0.5 * spatial_scale
      Dtype roi_end_h =   ( (pred_ctr_y + 0.5*(pred_h-1)) - 1 ) * spatial_scale; // dreh / dpcy = spatial_scale; dreh / dph =  0.5 * spatial_scale
      assert(roi_end_w - roi_start_w >= 0);
      assert(roi_end_h - roi_start_h >= 0);
  
      Dtype roi_width  = roi_end_w - roi_start_w + 1;
      Dtype roi_height = roi_end_h - roi_start_h + 1;
  
      Dtype bin_size_pw = static_cast<Dtype>(pooled_width)  / roi_width;  
      Dtype bin_size_ph = static_cast<Dtype>(pooled_height) / roi_height; 
  
      int pwstart = static_cast<int>(floor(static_cast<Dtype>(-roi_start_w + w) 
                                          * bin_size_pw)); 
      int phstart = static_cast<int>(floor(static_cast<Dtype>(-roi_start_h + h)
                                          * bin_size_ph)); 
      int pwend = static_cast<int>(ceil(static_cast<Dtype>(-roi_start_w + w + 1) 
                                       * bin_size_pw));
      int phend = static_cast<int>(ceil(static_cast<Dtype>(-roi_start_h + h + 1)  
                                       * bin_size_ph)); 
   
      //bottom_diff_data[index] = pwend; //phend; 
      //bottom_diff_data[index] = pwstart+1; //phend; 
  
      // Clip to top boundaries
      phstart = min(max(phstart, 0), pooled_height);         
      phend =   min(max(phend, 0),   pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend =   min(max(pwend, 0),   pooled_width);
  
      Dtype w_ = w, h_ = h; 
      Dtype wctr = 0, wdiff = 0, hctr = 0, hdiff = 0;
      Dtype gain = 0, gain_x = 0, gain_y = 0, gain_x_all = 0, gain_y_all = 0;  
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          int top_index = nth_roi * (channels * pooled_height * pooled_width) + c * (pooled_height * pooled_width) + ph * pooled_width  + pw;
          int top_buffer_index = nth_roi * (channels * pooled_height * pooled_width * 6) + c * (pooled_height * pooled_width * 6) + ph * (pooled_width * 6) + pw * 6;
          wctr       = top_data_buffer[top_buffer_index+0]; 
          wdiff      = top_data_buffer[top_buffer_index+1]; 
          hctr       = top_data_buffer[top_buffer_index+2]; 
          hdiff      = top_data_buffer[top_buffer_index+3]; 
          gain_x_all = top_data_buffer[top_buffer_index+4]; 
          gain_y_all = top_data_buffer[top_buffer_index+5]; 
  
          gain_x = (wdiff - abs((w_ - wctr))) / wdiff;   // dgx / dwdiff =   (w-wctr) / (wdiff)^2 ( if w >= wctr )
                                                         // dgx / dwdiff = - (w-wctr) / (wdiff)^2 ( else )
                                                         // dgx / dwctr  =   1 / wdiff ( if w >= wctr )
                                                         // dgx / dwctr  = - 1 / wdiff ( else )
          gain_y = (hdiff - abs((h_ - hctr))) / hdiff;   // dgy / dhdiff =   (h-hctr) / (hdiff)^2 ( if h >= hctr )
                                                                                                // dgy / dhdiff = - (h-hctr) / (hdiff)^2 ( else )
                                                                                                // dgy / dhctr  =   1 / hdiff ( if h >= hctr )
                                                                                                // dgy / dhdiff = - 1 / hdiff ( else )
          if (gain_x_all > 1e-10) 
            gain_x = gain_x / gain_x_all; 
          if (gain_y_all > 1e-10)  
            gain_y = gain_y / gain_y_all; 
  
          gain = gain_x * gain_y;
          bottom_diff_data[index] = bottom_diff_data[index] + gain * top_diff[top_index]; //val = val + gain * bottom_data[bottom_index];
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void ROIWarpBackwardDeltaROI(const int nthreads, const Dtype* top_data_buffer,
    const Dtype spatial_scale, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, 
    const Dtype* bottom_rois, const Dtype* bottom_delta_rois,
    const Dtype* top_diff,
    Dtype* bottom_diff_data,
    Dtype* bottom_diff_delta_rois_buffer) {
  CUDA_KERNEL_LOOP(index, nthreads) { 
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    int buffer_index = n * (channels * pooled_height * pooled_width * 6) + c * (pooled_height * pooled_width * 6) + ph * (pooled_width * 6) + pw * 6;
    Dtype gain_x_all = top_data_buffer[buffer_index+4];
    Dtype gain_y_all = top_data_buffer[buffer_index+5];

    bottom_rois += n * 5;
    int roi_batch_ind = (bottom_rois[0] - 1);

    Dtype src_w = bottom_rois[3] - bottom_rois[1] + 1; 
    Dtype src_h = bottom_rois[4] - bottom_rois[2] + 1;
    Dtype src_ctr_x = bottom_rois[1] + 0.5*(src_w-1.0); 
    Dtype src_ctr_y = bottom_rois[2] + 0.5*(src_h-1.0); 

    Dtype dst_ctr_x = bottom_delta_rois[1]; // dx (in fast-rcnn notation) = cx (in here)
    Dtype dst_ctr_y = bottom_delta_rois[2]; // dy (in fast-rcnn notation) = cy (in here) 
    Dtype dst_scl_x = bottom_delta_rois[3]; // dw (in fast-rcnn notation) = sx (in here)
    Dtype dst_scl_y = bottom_delta_rois[4]; // dh (in fast-rcnn notation) = sy (in here) 

    Dtype pred_ctr_x = dst_ctr_x * src_w + src_ctr_x; // dpcx / dcx = src_w
    Dtype pred_ctr_y = dst_ctr_y * src_h + src_ctr_y; // dpcy / dcy = src_h
    Dtype pred_w = exp(dst_scl_x) * src_w;            // dpw  / dsx = src_w * exp(dsx)  
    Dtype pred_h = exp(dst_scl_y) * src_h;            // dph  / dsy = src_h * exp(dsy)  
    
    Dtype roi_start_w = ( (pred_ctr_x - 0.5*(pred_w-1)) - 1 ) * spatial_scale; // drsw / dpcx = spatial_scale; drsw / dpw = -0.5 * spatial_scale
    Dtype roi_start_h = ( (pred_ctr_y - 0.5*(pred_h-1)) - 1 ) * spatial_scale; // drsh / dpcy = spatial_scale; drsh / dph = -0.5 * spatial_scale
    Dtype roi_end_w =   ( (pred_ctr_x + 0.5*(pred_w-1)) - 1 ) * spatial_scale; // drew / dpcx = spatial_scale; drew / dpw =  0.5 * spatial_scale
    Dtype roi_end_h =   ( (pred_ctr_y + 0.5*(pred_h-1)) - 1 ) * spatial_scale; // dreh / dpcy = spatial_scale; dreh / dph =  0.5 * spatial_scale
    assert(roi_end_w - roi_start_w >= 0); 
    assert(roi_end_h - roi_start_h >= 0); 
    
    // drsw / dcx = drsw / dpcx * dpcx / dcx = spatial_scale * src_w
    // drew / dcx = drew / dpcx * dpcx / dcx = spatial_scale * src_w

    // drsh / dcy = drsh / dpcy * dpcy / dcy = spatial_scale * src_h
    // dreh / dcy = dreh / dpcy * dpcy / dcy = spatial_scale * src_h

    // drsw / dsx = drsw / dpw * dpw / dsx = -0.5 * spatial_scale * src_w * exp(dsx) 
    // drew / dsx = drew / dpw * dpw / dsx =  0.5 * spatial_scale * src_w * exp(dsx)
 
    // drsh / dsy = drsh / dph * dph / dsy = -0.5 * spatial_scale * src_h * exp(dsy)
    // dreh / dsy = dreh / dph * dph / dsy =  0.5 * spatial_scale * src_h * exp(dsy) 
 
    // Force malformed ROIs to be 1x1
    Dtype roi_width  = roi_end_w - roi_start_w + 1; // drw / drew =  1 
                                                    // drw / drsw = -1
    Dtype roi_height = roi_end_h - roi_start_h + 1; // drh / dreh =  1 
                                                    // drh / drsh = -1 
    // drw / dcx = drw / drew * drew / dcx + drw / drsw * drsw / dcx = drew / dcx - drsw / dcx = spatial_scale * src_w - spatial_scale * src_w = 0 
    // drh / dcy = drh / dreh * dreh / dcy + drh / drsh * drsh / dcy = dreh / dcy - drsh / dcy = spatial_scale * src_h - spatial_scale * src_h = 0 
    // drw / dsx = drw / drew * drew / dsx + drw / drsw * drsw / dsx = drew / dsx - drsw / dsx = 0.5 * spatial_scale * src_w * exp(dsx) - (-0.5 * spatial_scale * src_w * exp(dsx)) = spatial_scale * src_w * exp(dsx) 
    // drh / dsy = drh / dreh * dreh / dsy + drh / drsh * drsh / dsy = dreh / dsy - drsh / dsy = 0.5 * spatial_scale * src_h * exp(dsy) - (-0.5 * spatial_scale * src_h * exp(dsy)) = spatial_scale * src_h * exp(dsy) 

    Dtype bin_size_w = roi_width  / static_cast<Dtype>(pooled_width);  // dbw / drw  =  1 / pooled_width
    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height); // dbh / drh  =  1 / pooled_height
    // dbw / dcx = dbw / drw * drw / dcx = 0 
    // dbh / dcy = dbh / drh * drh / dcy = 0
    // dbw / dsx = dbw / drw * drw / dsx = 1 / pooled_width * spatial_scale * src_w * exp(dsx) 
    // dbh / dsy = dbh / drh * drh / dsy = 1 / pooled_height * spatial_scale * src_h * exp(dsy) 

    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)        // dws / dbw = pw 
                                        * bin_size_w)) + roi_start_w; // dws / drsw = 1
    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)        // dhs / dbh = ph 
                                        * bin_size_h)) + roi_start_h; // dhs / drsh = 1 
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)       // dwe / dbw = (pw+1)
                                     * bin_size_w)) + roi_start_w;    // dwe / drsw = 1 
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)       // dhe / dbh = (ph+1)
                                     * bin_size_h)) + roi_start_h;    // dhe / drsh = 1 
    // dws / dcx = dws / dbw * dbw / dcx + dws / drsw * drsw / dcx = pw * 0 + 1 * spatial_scale * src_w     = spatial_scale * src_w
    // dwe / dcx = dwe / dbw * dbw / dcx + dwe / drsw * drsw / dcx = (pw+1) * 0 + 1 * spatial_scale * src_w = spatial_scale * src_w

    // dws / dsx = dws / dbw * dbw / dsx + dws / drsw * drsw / dsx = pw * 1 / pooled_width * spatial_scale * src_w * exp(dsx) + 1 * (-0.5) * spatial_scale * src_w * exp(dsx) = ( pw / pooled_width - 0.5 ) * spatial_scale * src_w * exp(dsx) 
    // dwe / dsx = dwe / dbw * dbw / dsx + dwe / drsw * drsw / dsx = (pw+1) * 1 / pooled_width * spatial_scale * src_w * exp(dsx) + 1 * 0.5 * spatial_scale * src_w * exp(dsx) = ( (pw+1)/pooled_width + 0.5 ) * spatial_scale * src_w * exp(dsx)

    // dhs / dcy = dhs / dbh * dbh / dcy + dhs / drsh * drsh / dcy = ph * 0 + 1 * spatial_scale * src_h     = spatial_scale * src_w
    // dhe / dcy = dhe / dbh * dbh / dcy + dhe / drsh * drsh / dcy = (ph+1) * 0 + 1 * spatial_scale * src_h = spatial_scale * src_h

    // dhs / dsy = dhs / dbh * dbh / dsy + dhs / drsh * drsh / dsy = ph * 1 / pooled_height * spatial_scale * src_h * exp(dsy) + 1 * (-0.5) * spatial_scale * src_h * exp(dsy) = (ph / pooled_height - 0.5) * spatial_scale * src_h * exp(dsy) 
    // dhe / dsy = dhe / dbh * dbh / dsy + dhe / drsh * drsh / dsy = (ph+1) * 1 / pooled_height * spatial_scale * src_h * exp(dsy) + 1 * 0.5 * spatial_scale * src_h * exp(dsy) = ((ph+1)/pooled_height + 0.5) * spatial_scale * src_h * exp(dsy)  

    Dtype wctr = static_cast<Dtype>(wend-1+wstart) * 0.5;      // dwctr / dwe = 0.5; dwctr / dws = 0.5 
    Dtype hctr = static_cast<Dtype>(hend-1+hstart) * 0.5;      // dhctr / dhe = 0.5; dhctr / dhs = 0.5 
    Dtype wdiff = max(static_cast<Dtype>(wend-1-wstart), 1.);  // dwdiff / dwe = (wend-wstart) >= 1 ? 1 : 0; dwdiff / dws = (wend-wstart) >= 1 ? -1 : 0; 
    Dtype hdiff = max(static_cast<Dtype>(hend-1-hstart), 1.);  // dhdiff / dhe = (hend-hstart) >= 1 ? 1 : 0; dhdiff / dhs = (hend-hstart) >= 1 ? -1 : 0;
    Dtype wdiff_mask = (wend-wstart) >= 1 ? 1 : 0;
    Dtype hdiff_mask = (wend-wstart) >= 1 ? 1 : 0;
    // dwctr / dcx = dwctr / dwe * dwe / dcx + dwctr / dws * dws / dcx = 0.5 * spatial_scale * src_w + 0.5 * spatial_scale * src_w = spatial_scale * src_w 
    // dwdiff / dcx = dwdiff / dwe * dwe / dcx + dwdiff / dws * dws / dcx = 1 * spatial_scale * src_w -  1  * spatial_scale * src_w = 0 

    // dhctr / dcy = spatial_scale * src_h
    // dhdiff / dcy = 0
  
    // dwctr / dsx = dwctr / dwe * dwe / dsx + dwctr / dws * dws / dsx = 0.5 * ((pw+1)/pooled_width + 0.5) * spatial_scale * src_w * exp(dsx) + 0.5 * (pw/pooled_width - 0.5) * spatial_scale * src_w * exp(dsx) 
    //                                                                 = 0.5 * (2*pw+1)/pooled_width * spatial_scale * src_w * exp(dsx)
    //                                                                 = (pw + 0.5) / pooled_width * spatial_scale * src_w * exp(dsx) 
    // dwdiff / dsx = dwdiff / dwe * dwe / dsx + dwdiff / dws * dws / dsx = 1 * ((pw+1)/pooled_width + 0.5) * spatial_scale * src_w * exp(dsx) + (-1) * (pw/pooled_width - 0.5) * spatial_scale * src_w * exp(dsx)
    //                                                                    = (wend-wstart) >= 1 ? (1 / pooled_width + 1) * spatial_scale * src_w * exp(dsx) : 0 
    // dhctr / dsy  = (ph + 0.5) / pooled_height * spatial_scale * src_h * exp(dsy)
    // dhdiff / dsy = (hend-hstart) >= 1 ? (1 / pooled_height + 1) * spatial_scale * src_h * exp(dsy) : 0
  
    // if w >= wctr  
    // dgx / dcx = dgx / dwctr * dwctr / dcx + dgx / dwdiff * dwdiff / dcx = 1 / wdiff * spatial_scale * src_w + (( w - wctr ) / (wdiff)^2 ) * 0
    //                                                                     = 1 / wdiff * spatial_scale * src_w  
    // dgx / dsx = dgx / dwctr * dwctr / dsx + dgx / dwdiff * dwdiff / dsx = 1 / wdiff * (pw + 0.5) / pooled_width * spatial_scale * src_w * exp(dsx) + ((wend-wstart) >= 1 ? 1 : 0) * (( w - wctr ) / (wdiff)^2 ) * (1 / pooled_width + 1) * spatial_scale * src_w * exp(dsx)
    //                                                                     = ((pw * 0.5) / (pooled_width * wdiff) + ((wend-wstart) >= 1 ? 1 : 0) * (( w - wctr ) / (wdiff)^2 ) * (1 + pooled_width) / pooled_width ) * spatial_scale * src_w * exp(dsx) 
    // dgy / dcy = dgy / dhctr * dhctr / dcy + dgy / dhdiff * dhdiff / dcy = 1 / hdiff * spatial_scale * src_h
    // dgy / dsy = dgy / dhctr * dhctr / dsy + dgy / dhdiff * dhdiff / dsy = ((ph * 0.5) / (pooled_height * hdiff) + ((hend-hstart) >= 1 ? 1 : 0) * (( h - hctr ) / (hdiff)^2 ) * (1 + pooled_height) / pooled_height ) * spatial_scale * src_h * exp(dsy) 
  
  
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);         //  
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);
    //bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype val_cx = 0, val_cy = 0, val_sx = 0, val_sy = 0; 
    Dtype gain = 0, gain_x = 0, gain_y = 0;  
    Dtype pw_ = static_cast<Dtype>(pw); 
    Dtype ph_ = static_cast<Dtype>(ph);
    Dtype pooled_width_  = static_cast<Dtype>(pooled_width); 
    Dtype pooled_height_ = static_cast<Dtype>(pooled_height);
    Dtype src_w_ = static_cast<Dtype>(src_w); 
    Dtype src_h_ = static_cast<Dtype>(src_h);  
    Dtype buffer_sx = 0, buffer_sy = 0;  
    //bottom_data += (roi_batch_ind * channels + c) * height * width;
    bottom_diff_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        Dtype w_ = w, h_ = h;  
        gain_x = (wdiff - abs((w_ - wctr))) / wdiff;   // dgx / dwdiff =   (w-wctr) / (wdiff)^2 ( if w >= wctr ) 
                                                       // dgx / dwdiff = - (w-wctr) / (wdiff)^2 ( else )
                                                       // dgx / dwctr  =   1 / wdiff ( if w >= wctr )  
                                                       // dgx / dwctr  = - 1 / wdiff ( else )  
        gain_y = (hdiff - abs((h_ - hctr))) / hdiff;   // dgy / dhdiff =   (h-hctr) / (hdiff)^2 ( if h >= hctr ) 
                                                                                              // dgy / dhdiff = - (h-hctr) / (hdiff)^2 ( else )
                                                                                              // dgy / dhctr  =   1 / hdiff ( if h >= hctr )
                                                                                              // dgy / dhdiff = - 1 / hdiff ( else )
        if (gain_x_all > 1e-10)
          gain_x = gain_x / gain_x_all;
        if (gain_y_all > 1e-10)
          gain_y = gain_y / gain_y_all;

        // buffer 
        Dtype coeff_x = w >= wctr ? 1 : -1; coeff_x = coeff_x * gain_y * spatial_scale * src_w_ * top_diff[index]; 
        Dtype coeff_y = h >= hctr ? 1 : -1; coeff_y = coeff_y * gain_x * spatial_scale * src_h_ * top_diff[index]; 
        val_cx = val_cx + coeff_x / wdiff;  
        val_cy = val_cy + coeff_y / hdiff;
        //val_sx = val_sx + coeff_x * (pw_ * 0.5 * wdiff + (w_ - wctr) * (1 + pooled_width_ )) / (wdiff*wdiff) / pooled_width_  * exp(dst_scl_x);
        //val_sy = val_sy + coeff_y * (ph_ * 0.5 * hdiff + (h_ - hctr) * (1 + pooled_height_)) / (hdiff*hdiff) / pooled_height_ * exp(dst_scl_y);
        buffer_sx = 0; buffer_sx = coeff_x * (pw_ * 0.5 * wdiff + wdiff_mask * (w_ - wctr) * (1 + pooled_width_ )); buffer_sx = buffer_sx / (wdiff*wdiff); buffer_sx = buffer_sx / pooled_width_  * exp(dst_scl_x);  
        val_sx = val_sx + buffer_sx; 
        buffer_sy = 0; buffer_sy = coeff_y * (ph_ * 0.5 * hdiff + hdiff_mask * (h_ - hctr) * (1 + pooled_height_)); buffer_sy = buffer_sy / (hdiff*hdiff); buffer_sy = buffer_sy / pooled_height_ * exp(dst_scl_y);
        val_sy = val_sy + buffer_sy; 
        //(dgain/ddelta_rois) * top_diff[index]; // dgain/ddeleta_rois = dgain/dgain_x * dgain_x/ddelta_rois + dgain/dgain_y * dgain_y/ddelta_rois
                                                 //                    =        gain_y * dgain_x/ddelta_rois +        gain_x * dgain_y/ddelta_rois
      }
    }
    /*int*/ buffer_index = n * (channels * pooled_height * pooled_width * 4) + c * (pooled_height * pooled_width * 4) + ph * (pooled_width * 4) + pw * 4; 
    bottom_diff_delta_rois_buffer[buffer_index+0] = val_cx; 
    bottom_diff_delta_rois_buffer[buffer_index+1] = val_cy; 
    bottom_diff_delta_rois_buffer[buffer_index+2] = val_sx;
    bottom_diff_delta_rois_buffer[buffer_index+3] = val_sy;
    //bottom_diff_delta_rois_cx[index] = val_cx;
    //bottom_diff_delta_rois_cy[index] = val_cy;
    //bottom_diff_delta_rois_sx[index] = val_sx;
    //bottom_diff_delta_rois_sy[index] = val_sy;
  }
}


extern "C"
void inn_ROIWarping_updateGradInputAtomic(THCState *state,
    THCudaTensor *gradInput_data, THCudaTensor *data,   
    THCudaTensor *gradInput_delta_rois, THCudaTensor *delta_rois,
    THCudaTensor *gradInput_delta_rois_buffer,
    THCudaTensor *gradOutput, THCudaTensor *top_data_buffer, 
    THCudaTensor* rois, int W, int H, double spatial_scale)
{
  THAssert(THCudaTensor_nDimension(state, data) == 4);
  THAssert(THCudaTensor_nDimension(state, top_data_buffer) == 5);
  THAssert(THCudaTensor_nDimension(state, rois) == 2 && rois->size[1] == 5);
  THAssert(THCudaTensor_nDimension(state, delta_rois) == 2 && delta_rois->size[1] == 5);
  THAssert(THCudaTensor_nDimension(state, rois) == THCudaTensor_nDimension(state, delta_rois) &&
           rois->size[0] == delta_rois->size[0] &&
           rois->size[1] == delta_rois->size[1]);
  THAssert(THCudaTensor_isContiguous(state, data));
  THAssert(THCudaTensor_isContiguous(state, top_data_buffer));
  THAssert(THCudaTensor_isContiguous(state, rois));
  THAssert(THCudaTensor_isContiguous(state, delta_rois));
  long num_rois = rois->size[0];
  long nInputPlane = data->size[1];
  THCudaTensor_resizeAs(state, gradInput_data, data);
  THCudaTensor_zero(state, gradInput_data);
  THCudaTensor_resizeAs(state, gradInput_delta_rois, delta_rois);
  THCudaTensor_zero(state, gradInput_delta_rois);
  THCudaTensor_resize5d(state, gradInput_delta_rois_buffer, num_rois, nInputPlane, H, W, 4);
  THCudaTensor_zero(state, gradInput_delta_rois_buffer);

  //Backpropagation for data
  long count = THCudaTensor_nElement(state, gradInput_data);
  for (int nth_roi = 0; nth_roi < num_rois; ++nth_roi) {
    ROIWarpBackwardData<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS / 2, 0, THCState_getCurrentStream(state)>>>(
        count,
        THCudaTensor_data(state, top_data_buffer),
        spatial_scale, nInputPlane, data->size[2], data->size[3], H, W, nth_roi,
        THCudaTensor_data(state, rois),
        THCudaTensor_data(state, delta_rois),
        THCudaTensor_data(state, gradOutput), 
        THCudaTensor_data(state, gradInput_data),
        THCudaTensor_data(state, gradInput_delta_rois_buffer)
        );
  }

  //Backpropagation for delta_roi
  count = THCudaTensor_nElement(state, gradOutput);
  ROIWarpBackwardDeltaROI<float><<<GET_BLOCKS(count), CUDA_NUM_THREADS / 2, 0, THCState_getCurrentStream(state)>>>(
      count,
      THCudaTensor_data(state, top_data_buffer),
      spatial_scale, nInputPlane, data->size[2], data->size[3], H, W, 
      THCudaTensor_data(state, rois),
      THCudaTensor_data(state, delta_rois),
      THCudaTensor_data(state, gradOutput),
      THCudaTensor_data(state, gradInput_data),
      THCudaTensor_data(state, gradInput_delta_rois_buffer)
      );

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in inn_ROIWarping_updateGradInputAtomic: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
}
