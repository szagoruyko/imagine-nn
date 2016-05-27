--[[
  This code is borrowed from AffineGridGeneratorBHWD.lua in https://github.com/qassemoquab/stnbhwd
]]

local RWGG, parent = torch.class('inn.ROIWarpingGridGenerator', 'nn.Container')

local function fast_rcnn_bbox_transform_inv(rois, delta_rois)
-- rois       : N by 4 torch.Tensor; for each row, rois[{n, {}}]       == x_start, y_start, x_end, y_end (in image coordinates)
-- delta_rois : N by 4 torch.Tensor; for each row, delta_rois[{n, {}}] == dx, dy, dw, dh (in fast-rcnn notation)

  local src_w = rois[{{},3}] - rois[{{},1}] + 1;
  local src_h = rois[{{},4}] - rois[{{},2}] + 1;
  local src_ctr_x = rois[{{},1}] + 0.5*(src_w-1.0);
  local src_ctr_y = rois[{{},2}] + 0.5*(src_h-1.0);

  local dst_ctr_x = delta_rois[{{},1}]; -- dx (in fast-rcnn notation)
  local dst_ctr_y = delta_rois[{{},2}]; -- dy (in fast-rcnn notation)
  local dst_scl_x = delta_rois[{{},3}]; -- dw (in fast-rcnn notation)
  local dst_scl_y = delta_rois[{{},4}]; -- dh (in fast-rcnn notation)

  local pred_ctr_x = torch.cmul(dst_ctr_x, src_w) + src_ctr_x;
  local pred_ctr_y = torch.cmul(dst_ctr_y, src_h) + src_ctr_y;
  local pred_w = torch.cmul(torch.exp(dst_scl_x), src_w);
  local pred_h = torch.cmul(torch.exp(dst_scl_y), src_h);

  local roi_start_w = pred_ctr_x - 0.5*(pred_w-1);
  local roi_start_h = pred_ctr_y - 0.5*(pred_h-1);
  local roi_end_w =   pred_ctr_x + 0.5*(pred_w-1);
  local roi_end_h =   pred_ctr_y + 0.5*(pred_h-1);

  return torch.cat({roi_start_w, roi_start_h, roi_end_w, roi_end_h}, 2)
end

--function RWGG:__init(height, width, spatial_scale)
function RWGG:__init(height, width)
   parent.__init(self)
   assert(height > 1)
   assert(width > 1)
   self.height = height
   self.width = width

   self.output_tmp = {}
   self.gradInput = {}

   --self.spatial_scale = spatial_scale or 1

   self.baseGrid = torch.Tensor(self.height, self.width, 2) -- Grid for input image
   for i=1,self.width do
      self.baseGrid:select(3,1):select(2,i):fill(i-1)
   end
   for j=1,self.height do
      self.baseGrid:select(3,2):select(1,j):fill(j-1)
   end
   self.batchGrid = torch.Tensor(1, height, width, 2):copy(self.baseGrid)
end

function RWGG:updateOutput(input) --(_transformMatrix)
  assert(#input == 1 or #input == 2)
  local rois = input[1]
  local delta_rois
  if #input == 2 then
    delta_rois = input[2]
  else -- #input == 2
    self.delta_rois = self.delta_rois or rois.new()
    self.delta_rois:resizeAs(rois):zero()
    self.delta_rois[{{}, 1}] = rois[{{}, 1}]
    delta_rois = self.delta_rois
  end
  assert(rois:dim() == 2 and delta_rois:dim() == 2)
  assert(rois:size(2) == 5 and delta_rois:size(2) == 5)

  local batch_size = rois:size(1)

  if self.batchGrid:size(1) ~= batch_size then
     self.batchGrid:resize(batch_size, self.height, self.width, 2)
     for i=1,batch_size do
        self.batchGrid:select(1,i):copy(self.baseGrid)
     end
  end

  -- allocate output
  self.output_tmp[1] = self.output_tmp[1] or rois.new()
  self.output_tmp[2] = self.output_tmp[2] or rois.new()
  local grid_ctrs = self.output_tmp[1] 
  local bin_sizes = self.output_tmp[2]
 
  -- prepare msc 
  local pred_rois = fast_rcnn_bbox_transform_inv(rois[{{}, {2, 5}}], delta_rois[{{}, {2, 5}}])

  local rois_width  = pred_rois[{{}, 3}] - pred_rois[{{}, 1}]
  local rois_height = pred_rois[{{}, 4}] - pred_rois[{{}, 2}]
  local rois_start_width = pred_rois[{{}, 1}]
  local rois_start_height = pred_rois[{{}, 2}]

  local bin_size_w = rois_width  / self.width 
  local bin_size_h = rois_height / self.height
 
  grid_ctrs:resize(batch_size, self.height, self.width, 2):fill(0) -- b x h x w x 2 (x, y == width, height)
  bin_sizes:resize(batch_size, 2):fill(0)                          -- b x 2         (x, y == width, height) 

  -- update bin_sizes
  bin_sizes:select(2,1):copy(bin_size_w:reshape(batch_size, 1)) -- width 
  bin_sizes:select(2,2):copy(bin_size_h:reshape(batch_size, 1)) -- height

  -- update grid_ctrs 
  local grid_ctrs_w = grid_ctrs[{{}, {}, {}, {1}}] -- allocate address 
  grid_ctrs_w:copy(self.batchGrid[{{}, {}, {}, {1}}])
             :cmul(bin_size_w:reshape(batch_size, 1, 1, 1)
                             :expand(batch_size, self.height, self.width, 1))
             :add(bin_size_w:reshape(batch_size, 1, 1, 1)
                            :expand(batch_size, self.height, self.width, 1) / 2)
             :add(rois_start_width:reshape(batch_size, 1, 1, 1)
                                  :expand(batch_size, self.height, self.width, 1))
  local grid_ctrs_h = grid_ctrs[{{}, {}, {}, {2}}] -- allocate address
  grid_ctrs_h:copy(self.batchGrid[{{}, {}, {}, {2}}])
             :cmul(bin_size_h:reshape(batch_size, 1, 1, 1)
                             :expand(batch_size, self.height, self.width, 1))
             :add(bin_size_h:reshape(batch_size, 1, 1, 1)
                            :expand(batch_size, self.height, self.width, 1) / 2)
             :add(rois_start_height:reshape(batch_size, 1, 1, 1)
                                   :expand(batch_size, self.height, self.width, 1))

  return self.output_tmp
end

function RWGG:updateGradInput(input, gradOutput) --(_transformMatrix, _gradGrid)
  assert(#input == 1 or #input == 2)
  local rois = input[1]
  local delta_rois
  if #input == 2 then
    delta_rois = input[2]
  else -- #input == 2
    self.delta_rois = self.delta_rois or rois.new()
    self.delta_rois:resizeAs(rois):zero()
    self.delta_rois[{{}, 1}] = rois[{{}, 1}]
    delta_rois = self.delta_rois
  end
  assert(rois:dim() == 2 and delta_rois:dim() == 2)
  assert(rois:size(2) == 5 and delta_rois:size(2) == 5)

  local batch_size = rois:size(1)

  self.batchGrid = self.batchGrid:typeAs(rois)
  self.baseGrid = self.baseGrid:typeAs(rois)

  if self.batchGrid:size(1) ~= batch_size then
     self.batchGrid:resize(batch_size, self.height, self.width, 2)
     for i=1,batch_size do
        self.batchGrid:select(1,i):copy(self.baseGrid)
     end
  end

  -- init output buffer
  self.gradInput_rois = self.gradInput_rois or rois.new()
  self.gradInput_delta_rois = self.gradInput_delta_rois or delta_rois.new()
  self.gradInput_rois:resizeAs(rois):zero()
  self.gradInput_delta_rois:resizeAs(delta_rois):zero()

  -- prepare msc 
  --local pred_rois = fast_rcnn_bbox_transform_inv(rois[{{}, {2, 5}}], delta_rois[{{}, {2, 5}}])

  --local rois_width  = pred_rois[{{}, 3}] - pred_rois[{{}, 1}]
  --local rois_height = pred_rois[{{}, 4}] - pred_rois[{{}, 2}]

  --local bin_size_w = rois_width  / self.width
  --local bin_size_h = rois_height / self.height

  local src_width  = rois[{{}, {4}}] - rois[{{}, {2}}] + 1; src_width  = src_width:reshape(batch_size, 1, 1)
  local src_height = rois[{{}, {5}}] - rois[{{}, {3}}] + 1; src_height = src_height:reshape(batch_size, 1, 1)

  local flattenedBatchGrid = self.batchGrid:view(batch_size, self.width*self.height, 2)

  -- grad from grid_ctrs 

  -- drsw / dcx = drsw / dpcx * dpcx / dcx = spatial_scale * src_w
  -- drew / dcx = drew / dpcx * dpcx / dcx = spatial_scale * src_w
  -- drsh / dcy = drsh / dpcy * dpcy / dcy = spatial_scale * src_h
  -- dreh / dcy = dreh / dpcy * dpcy / dcy = spatial_scale * src_h

  -- drsw / dsx = drsw / dpw * dpw / dsx = -0.5 * spatial_scale * src_w * exp(dsx)
  -- drew / dsx = drew / dpw * dpw / dsx =  0.5 * spatial_scale * src_w * exp(dsx)
  -- drsh / dsy = drsh / dph * dph / dsy = -0.5 * spatial_scale * src_h * exp(dsy)
  -- dreh / dsy = dreh / dph * dph / dsy =  0.5 * spatial_scale * src_h * exp(dsy)

  -- grid_ctr_w = rsw + bin_size_w / 2 + pw * bin_size_w
  --            = rsw + (0.5 + pw) * bin_size_w 
  --            = rsw + (0.5 + pw) * (rew - rsw) / self.width
  --            = f(rsw, rew)
  -- dwctr / dcx = dwctr / drsw * drsw / dcx + dwctr / drew * drew / dcx
  --             = (1 + (0.5 + pw) / self.width * (-1)) * src_w  + ((0.5 + pw) / self.width * 1) * src_w
  --             = spatial_scale * src_w 
  -- dhctr / dcy = spatial_scale * src_h
  -- dwctr / dsx = dwctr / drsw * drsw / dsx + dwctr / drew * drew / dsx  
  --             = (1 + (0.5 + pw) / self.width * (-1)) * (-0.5 * spatial_scale * src_w * exp(dsx))
  --             + (    (0.5 + pw) / self.width *   1 ) * ( 0.5 * spatial_scale * src_w * exp(dsx))
  --             = (-1 + (0.5 + pw) / self.width * 2) * 0.5 * spatial_scale * src_w * exp(dsx)
  --             = 0.5 * spatial_scale * src_w * exp(dsx) * (-1 + 2 * (0.5 + pw) / self.width)  
  --             = ((pw + 0.5) / self.width  - 0.5) * spatial_scale * src_w * exp(dsx)
  -- dhctr / dsy = ((ph + 0.5) / self.height - 0.5) * spatial_scale * src_h * exp(dsy)

  -- grad from bin_sizes
  
  -- dbw / dcx = dbw / drw * drw / dcx = 0
  -- dbh / dcy = dbh / drh * drh / dcy = 0
  -- dbw / dsx = dbw / drw * drw / dsx = 1 / self.width  * spatial_scale * src_w * exp(dsx)
  -- dbh / dsy = dbh / drh * drh / dsy = 1 / self.height * spatial_scale * src_h * exp(dsy)

  local flattened_grid_ctrs = gradOutput[1]:view(batch_size, self.height*self.width, 2) -- b x ph x pw x 2
  local flattened_bin_sizes = gradOutput[2]                                             -- b x 2
  local flattened_gradInput_delta_rois = self.gradInput_delta_rois[{{},{2, 5}}]         -- b x 4

  flattened_gradInput_delta_rois[{{}, {1}}]:copy(torch.sum(torch.cmul( src_width:expand(batch_size, self.height * self.width, 1), flattened_grid_ctrs[{{}, {}, {1}}]), 2):reshape(batch_size, 1))
  flattened_gradInput_delta_rois[{{}, {2}}]:copy(torch.sum(torch.cmul(src_height:expand(batch_size, self.height * self.width, 1), flattened_grid_ctrs[{{}, {}, {2}}]), 2):reshape(batch_size, 1))
  flattened_gradInput_delta_rois[{{}, {3}}]:copy(torch.sum(torch.cmul( ((flattenedBatchGrid[{{}, {}, {1}}] + 0.5) / self.width - 0.5),  
                                                                       flattened_grid_ctrs[{{}, {}, {1}}]), 2):reshape(batch_size, 1))
  flattened_gradInput_delta_rois[{{}, {3}}]:add(torch.sum(torch.mul(flattened_bin_sizes[{{}, {1}}], 1/self.width), 2))
                                           :cmul(torch.exp(delta_rois[{{}, {4}}]))
                                           :cmul(src_width)
  flattened_gradInput_delta_rois[{{}, {4}}]:copy(torch.sum(torch.cmul( ((flattenedBatchGrid[{{}, {}, {2}}] + 0.5) / self.height - 0.5),  
                                                                       flattened_grid_ctrs[{{}, {}, {2}}]), 2):reshape(batch_size, 1))
  flattened_gradInput_delta_rois[{{}, {4}}]:add(torch.sum(torch.mul(flattened_bin_sizes[{{}, {2}}], 1/self.height), 2))
                                           :cmul(torch.exp(delta_rois[{{}, {5}}]))
                                           :cmul(src_height)
 
  -- update output
  self.gradInput[1] = self.gradInput_rois 
  self.gradInput[2] = self.gradInput_delta_rois

  return self.gradInput
end
