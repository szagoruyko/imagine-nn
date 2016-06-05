local ROIWarpingBilinearSample,parent = torch.class('inn.ROIWarpingBilinearSample', 'nn.Module')
local C = inn.C

local buffer_numbers = 6

--function ROIWarpingBilinearSample:__init(height, width, spatial_scale)
function ROIWarpingBilinearSample:__init(height, width)
  parent.__init(self)
  assert(width and height, 'height and width have to be provided')
  self.width = width
  self.height = height
  --self.spatial_scale = spatial_scale or 1
  self.gradInput = {}
end

--function ROIWarpingBilinearSample:setSpatialScale(scale)
--  self.spatial_scale = scale
--  return self
--end

function ROIWarpingBilinearSample:updateOutput(input)
  assert(#input == 4)
  local data = input[1]
  local grid_ctrs = input[2]
  local bin_sizes = input[3] 
  local roi_batch_inds = input[4] 

  local num_rois = roi_batch_inds:size(1)
  local nchannels = data:size(2)

  assert(grid_ctrs:size(1) == num_rois and 
         grid_ctrs:size(2) == self.height and
         grid_ctrs:size(3) == self.width and 
         grid_ctrs:size(4) == 2)
  assert(bin_sizes:size(1) == num_rois and 
         bin_sizes:size(2) == 2)
  
  self.output = self.output or data.new()
  self.output:resize(num_rois, nchannels, self.height, self.width):fill(0)
  self.output_buffer = self.output_buffer or data.new()
  self.output_buffer:resize(num_rois, self.height, self.width, buffer_numbers):fill(0)
 
  C.inn_ROIWarpingBilinearSample_updateOutput(cutorch.getState(),
    self.output:cdata(), self.output_buffer:cdata(),
    data:cdata(), grid_ctrs:cdata(), bin_sizes:cdata(), roi_batch_inds:cdata(), 
    self.width, self.height
    )--, self.spatial_scale)

  return self.output
end

function ROIWarpingBilinearSample:updateGradInput(input,gradOutput)
  assert(#input == 4)
  local data = input[1]
  local grid_ctrs = input[2]
  local bin_sizes = input[3]
  local roi_batch_inds = input[4] 

  local batch_size = data:size(1)
  local num_rois = roi_batch_inds:size(1)
  local nchannels = data:size(2)

  assert(self.output_buffer)
  assert(self.output_buffer:size(1) == num_rois and 
         self.output_buffer:size(2) == self.height and 
         self.output_buffer:size(3) == self.width and
         self.output_buffer:size(4) == buffer_numbers)

  self.gradInput_data = self.gradInput_data or data.new()                               -- b x c x h x w
  self.gradInput_grid_ctrs = self.gradInput_grid_ctrs or grid_ctrs.new()                -- n x h x w x 2
  self.gradInput_grid_ctrs_buffer = self.gradInput_grid_ctrs_buffer or grid_ctrs.new()  -- n x c x h x w x 2
  self.gradInput_bin_sizes = self.gradInput_bin_sizes or bin_sizes.new()                -- n x 2
  self.gradInput_bin_sizes_buffer = self.gradInput_bin_sizes_buffer or bin_sizes.new()  -- n x c x h x w x 2
  self.gradInput_roi_batch_inds = self.gradInput_roi_batch_inds or roi_batch_inds.new() -- n x 2

  self.gradInput_data:resizeAs(data):fill(0)
  self.gradInput_grid_ctrs:resizeAs(grid_ctrs):fill(0)
  self.gradInput_grid_ctrs_buffer:resize(num_rois, nchannels, self.height, self.width, 2):fill(0)
  self.gradInput_bin_sizes:resizeAs(bin_sizes):fill(0)
  self.gradInput_bin_sizes_buffer:resize(num_rois, nchannels, self.height, self.width, 2):fill(0)
  self.gradInput_roi_batch_inds:resize(num_rois, 2):fill(0)

  --print(self.output_buffer:select(4,1))
  --print(self.output_buffer:select(4,2))
  --print(self.output_buffer:select(4,3))
  --print(self.output_buffer:select(4,4))

  C.inn_ROIWarpingBilinearSample_updateGradInput(cutorch.getState(),
    self.gradInput_data:cdata(),      data:cdata(),
    self.gradInput_grid_ctrs:cdata(), grid_ctrs:cdata(), self.gradInput_grid_ctrs_buffer:cdata(), 
    self.gradInput_bin_sizes:cdata(), bin_sizes:cdata(), self.gradInput_bin_sizes_buffer:cdata(),
    roi_batch_inds:cdata(),
    self.output_buffer:cdata(), 
    gradOutput:cdata(),
    self.height, self.width
    ) --, self.spatial_scale)

 --print(self.gradInput_bin_sizes_buffer)

  self.gradInput_grid_ctrs:copy(self.gradInput_grid_ctrs_buffer:sum(2):view(num_rois, self.height, self.width, 2))
  self.gradInput_bin_sizes:copy(self.gradInput_bin_sizes_buffer:sum(2):sum(3):sum(4):view(num_rois, 2))
 
  --print(self.gradInput_grid_ctrs_buffer:select(2, 1):select(4, 1))
  ----print(self.gradInput_grid_ctrs_buffer:select(2, 1):select(4, 2))
  --print(self.gradInput_grid_ctrs_buffer:select(2, 2):select(4, 1))
  --print(self.gradInput_grid_ctrs_buffer:select(2, 3):select(4, 1))
  --for c = 1, nchannels do 
  --  print(self.gradInput_grid_ctrs_buffer:select(2, c))
  --end
  
  self.gradInput[1] = self.gradInput_data
  self.gradInput[2] = self.gradInput_grid_ctrs
  self.gradInput[3] = self.gradInput_bin_sizes
  self.gradInput[4] = self.gradInput_roi_batch_inds
 
  return self.gradInput 
end

function ROIWarpingBilinearSample:clearState()
   nn.utils.clear(self, 'gradInput_data', 'gradInput_grid_ctrs', 'gradInput_grid_ctrs_buffer', 'gradInput_bin_sizes', 'gradInput_bin_sizes_buffer', 'gradInput_roi_batch_inds')
   return parent.clearState(self)
end
