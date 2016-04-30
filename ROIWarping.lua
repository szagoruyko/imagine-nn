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
    self.delta_rois = self.delta_rois or rois.new()
    self.delta_rois:resizeAs(rois):zero()
    self.delta_rois[{{}, 1}] = rois[{{}, 1}] 
    --delta_rois = rois:clone()
    --delta_rois[{{},{2,5}}] = 0
    delta_rois = self.delta_rois
  end
  
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
    self.delta_rois = self.delta_rois or data.new()
    self.delta_rois:resizeAs(rois):zero()
    self.delta_rois[{{}, 1}] = rois[{{}, 1}]
    delta_rois = self.delta_rois
  end

  self.gradInput_boxes = self.gradInput_boxes or data.new()
  self.gradInput_rois = self.gradInput_rois or rois.new()
  self.gradInput_delta_rois = self.gradInput_delta_rois or delta_rois.new()
  --if #input == 3 then 
    --self.gradInput_delta_rois_dx = self.gradInput_delta_rois_dx or self.output.new()
    --self.gradInput_delta_rois_dy = self.gradInput_delta_rois_dy or self.output.new()
    --self.gradInput_delta_rois_dw = self.gradInput_delta_rois_dw or self.output.new()
    --self.gradInput_delta_rois_dh = self.gradInput_delta_rois_dh or self.output.new()
    self.gradInput_delta_rois_buffer = self.gradInput_delta_rois_buffer or self.output.new()
  --end 

  --C.inn_ROIWarping_updateGradInputAtomic(cutorch.getState(),
  --  self.gradInput_boxes:cdata(), self.indices:cdata(), data:cdata(),
  --  gradOutput:cdata(), rois:cdata(), self.W, self.H, self.spatial_scale)
  C.inn_ROIWarping_updateGradInputAtomic(cutorch.getState(),
    self.gradInput_boxes:cdata(), data:cdata(),
    self.gradInput_delta_rois:cdata(), delta_rois:cdata(),
    --self.gradInput_delta_rois_dx:cdata(),  
    --self.gradInput_delta_rois_dy:cdata(),  
    --self.gradInput_delta_rois_dw:cdata(),  
    --self.gradInput_delta_rois_dh:cdata(),  
    self.gradInput_delta_rois_buffer:cdata(),  
    gradOutput:cdata(), rois:cdata(), self.W, self.H, self.spatial_scale)

  --print(self.gradInput_delta_rois_buffer[{1, 1, 1, 1, {}}])

  self.gradInput_rois:resizeAs(rois):zero()

  self.gradInput_delta_rois[{{}, {2, 5}}] = self.gradInput_delta_rois_buffer:sum(2):sum(3):sum(4):view(rois:size()[1], 4)

  self.gradInput = {self.gradInput_boxes, self.gradInput_rois, self.gradInput_delta_rois}

  return self.gradInput
end

function ROIWarping:clearState()
   nn.utils.clear(self, 'gradInput_rois', 'gradInput_boxes', 'gradInput_delta_rois', 'gradInput_delta_rois_dx', 'gradInput_delta_rois_dy', 'gradInput_delta_rois_sx', 'gradInput_delta_rois_sy')
   return parent.clearState(self)
end
