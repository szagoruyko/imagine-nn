local ROIPooling,parent = torch.class('inn.ROIPooling', 'nn.Module')
local C = inn.C

function ROIPooling:__init(W,H,spatial_scale)
  parent.__init(self)
  assert(W and H, 'W and H have to be provided')
  self.W = W
  self.H = H
  self.spatial_scale = spatial_scale or 1
  self.gradInput = {}
  self.indices = torch.Tensor()
  self.v2 = true
end

function ROIPooling:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function ROIPooling:updateOutput(input)
  assert(#input == 2)
  local data = input[1]
  local rois = input[2]

  if self.v2 then
    C.inn_ROIPooling_updateOutputV2(cutorch.getState(),
      self.output:cdata(), self.indices:cdata(), data:cdata(), rois:cdata(),
      self.W, self.H, self.spatial_scale)
  else
    C.inn_ROIPooling_updateOutput(cutorch.getState(),
      self.output:cdata(), self.indices:cdata(), data:cdata(), rois:cdata(),
      self.W, self.H, self.spatial_scale)
  end
  return self.output
end

function ROIPooling:updateGradInput(input,gradOutput)
  local data = input[1]
  local rois = input[2]

  self.gradInput_boxes = self.gradInput_boxes or data.new()
  self.gradInput_rois = self.gradInput_rois or data.new()

  C.inn_ROIPooling_updateGradInputAtomic(cutorch.getState(),
    self.gradInput_boxes:cdata(), self.indices:cdata(), data:cdata(),
    gradOutput:cdata(), rois:cdata(), self.W, self.H, self.spatial_scale)

  self.gradInput_rois:resizeAs(rois):zero()

  self.gradInput = {self.gradInput_boxes, self.gradInput_rois}

  return self.gradInput
end

function ROIPooling:clearState()
   nn.utils.clear(self, 'gradInput_rois', 'gradInput_boxes', 'indices')
   return parent.clearState(self)
end
