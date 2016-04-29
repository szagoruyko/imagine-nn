local ROIWarping,parent = torch.class('inn.ROIWarping', 'nn.Module')
local C = inn.C

function ROIWarping:__init(W,H,spatial_scale)
  parent.__init(self)
  assert(W and H, 'W and H have to be provided')
  self.W = W
  self.H = H
  self.spatial_scale = spatial_scale or 1
  self.gradInput = {}
  --self.indices = torch.Tensor()
end

function ROIWarping:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function ROIWarping:updateOutput(input)
  assert(#input == 2 or #input == 3)
  local data = input[1]
  local rois = input[2]
  local delta_rois 
  if #input == 3 then
    delta_rois = input[3]
  else -- #input == 2  
    delta_rois = rois:clone()
    --delta_rois[{{}, 2}] = rois[{{},2}] + (rois[{{}, 4}] - rois[{{},2}]) / 2
    --delta_rois[{{}, 3}] = rois[{{},3}] + (rois[{{}, 5}] - rois[{{},3}]) / 2
    delta_rois[{{},{2,3}}] = 0 
    delta_rois[{{},{4,5}}] = 1
  end
  --print('hi')
  --print(rois)
  --print(delta_rois)
  
  --C.inn_ROIWarping_updateOutput(cutorch.getState(),
  --  self.output:cdata(), self.indices:cdata(), data:cdata(), rois:cdata(), delta_rois:cdata(), 
  --  self.W, self.H, self.spatial_scale)
  C.inn_ROIWarping_updateOutput(cutorch.getState(),
    self.output:cdata(), data:cdata(), rois:cdata(), delta_rois:cdata(), 
    self.W, self.H, self.spatial_scale)

  return self.output
end

function ROIWarping:updateGradInput(input,gradOutput)
  local data = input[1]
  local rois = input[2]
  local delta_rois
  if #input == 3 then
    delta_rois = input[3]
  else -- #input == 2
    delta_rois = rois:clone()
    delta_rois[{{},{2,3}}] = 0
    delta_rois[{{},{4,5}}] = 1
  end

  self.gradInput_boxes = self.gradInput_boxes or data.new()
  self.gradInput_rois = self.gradInput_rois or data.new()
  if #input == 3 then 
    self.gradInput_delta_rois = self.gradInput_delta_rois or data.new()
  end 

  --C.inn_ROIWarping_updateGradInputAtomic(cutorch.getState(),
  --  self.gradInput_boxes:cdata(), self.indices:cdata(), data:cdata(),
  --  gradOutput:cdata(), rois:cdata(), self.W, self.H, self.spatial_scale)

  self.gradInput_rois:resizeAs(rois):zero()

  self.gradInput = {self.gradInput_boxes, self.gradInput_rois, self.gradInput_delta_rois}

  return self.gradInput
end

function ROIWarping:clearState()
   nn.utils.clear(self, 'gradInput_rois', 'gradInput_boxes', 'indices')
   return parent.clearState(self)
end
