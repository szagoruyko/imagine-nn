local ROIPooling,parent = torch.class('inn.ROIPooling', 'nn.Module')
local C = inn.C

function ROIPooling:__init(W,H,spatial_scale)
  parent.__init(self)
  assert(W and H, 'W and H have to be provided')
  self.W = W
  self.H = H
  self.spatial_scale = spatial_scale or 1
  self.gradInput = {torch.Tensor()}
  self.indices = torch.Tensor()
end

function ROIPooling:setSpatialScale(scale)
  self.spatial_scale = spatial_scale
  return self
end

function ROIPooling:updateOutput(input)
  local data = input[1]
  local rois = input[2]

  C.inn_ROIPooling_updateOutput(cutorch.getState(),
    self.output:cdata(), self.indices:cdata(), data:cdata(), rois:cdata(),
    self.W, self.H, self.spatial_scale)
  return self.output
end

function ROIPooling:updateGradInput(input,gradOutput)
  local data = input[1]
  local rois = input[2]

  C.inn_ROIPooling_updateGradInput(cutorch.getState(),
    self.gradInput[1]:cdata(), self.indices:cdata(), data:cdata(),
    gradOutput:cdata(), rois:cdata(), self.W, self.H, self.spatial_scale)
  return self.gradInput
end

function ROIPooling:type(type)
  parent.type(self,type)
  self.gradInput[1] = self.gradInput[1]:type(type)
end
