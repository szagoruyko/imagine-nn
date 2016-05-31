local ROIWarping,parent = torch.class('inn.ROIWarping', 'nn.Module')
local C = inn.C

--function ROIWarping:__init(W,H,spatial_scale)
function ROIWarping:__init(H,W)
  parent.__init(self)
  assert(W and H, 'W and H have to be provided')
  self.W = W
  self.H = H
  --self.spatial_scale = spatial_scale or 1

  self.grid_gen = inn.ROIWarpingGridGenerator(self.H, self.W)
  self.sample  = inn.ROIWarpingBilinearSample(self.H, self.W)
  
  self.gradInput = {}
end

--function ROIWarping:setSpatialScale(scale)
--  self.spatial_scale = scale
--  return self
--end

function ROIWarping:updateOutput(input)
  assert(#input == 2 or #input == 3)
  local data = input[1]
  local rois = input[2]
  local delta_rois 
  if #input == 3 then
    delta_rois = input[3]
  else -- #input == 2 
    self.delta_rois = self.delta_rois or rois.new()
    self.delta_rois:resizeAs(rois):zero()
    self.delta_rois[{{}, 1}] = rois[{{}, 1}] 
    delta_rois = self.delta_rois
  end

  if torch.type(data) == 'torch.CudaTensor' then 
    self.grid_gen:cuda()
    self.sample:cuda()
  end

  self.grid_gen:updateOutput({rois, delta_rois})
  self.sample:updateOutput({data, self.grid_gen.output_tmp[1], self.grid_gen.output_tmp[2], self.grid_gen.output_tmp[3]})

  self.output = self.sample.output

  return self.output
end

function ROIWarping:updateGradInput(input,gradOutput)
  local data = input[1]
  local rois = input[2]
  local delta_rois
  if #input == 3 then
    delta_rois = input[3]
  else -- #input == 2
    self.delta_rois = self.delta_rois or data.new()
    self.delta_rois:resizeAs(rois):zero()
    self.delta_rois[{{}, 1}] = rois[{{}, 1}]
    delta_rois = self.delta_rois
  end

  if torch.type(data) == 'torch.CudaTensor' then
    self.grid_gen:cuda()
    self.sample:cuda()
  end

  self.sample:updateGradInput({data, self.grid_gen.output_tmp[1], self.grid_gen.output_tmp[2], self.grid_gen.output_tmp[3]}, gradOutput)
  self.grid_gen:updateGradInput({rois, delta_rois}, {self.sample.gradInput[2], self.sample.gradInput[3]})

  self.gradInput = {self.sample.gradInput[1], self.grid_gen.gradInput[1], self.grid_gen.gradInput[2]}

  return self.gradInput
end
