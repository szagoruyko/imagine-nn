local SpatialPyramidPooling, parent = torch.class('inn.SpatialPyramidPooling', 'nn.Module')

function SpatialPyramidPooling:__init(pyr)
  parent.__init(self)
  
  self.pyr = pyr
  
  self.dimd = 1
  self.module = nn.Concat(self.dimd)
  for i=1,#self.pyr do
    local t = nn.Sequential()
    t:add(nn.SpatialAdaptiveMaxPooling(self.pyr[i][1],self.pyr[i][2]))
    t:add(nn.View(-1):setNumInputDims(3))
    self.module:add(t)
  end
  
  self.output = torch.Tensor()
  
end

function SpatialPyramidPooling:updateOutput(input)
  
  local dimd = 1
  if input:nDimension() == 4 then 
    dimd = dimd + 1
  end

  if self.dimd ~= dimd then
    self.dimd = dimd
    self.module.dimension = dimd
  end
  
  assert(input:type()==self.output:type(),'Wrong input type!')
  
  self.output = self.module:updateOutput(input)
  return self.output
end

function SpatialPyramidPooling:updateGradInput(input, gradOutput)
  
  local dimd = 1
  if input:nDimension() == 4 then 
    dimd = dimd + 1
  end

  if self.dimd ~= dimd then
    self.dimd = dimd
    self.module.dimension = dimd
  end
  
  assert(input:type()==self.gradInput:type(),'Wrong input type!')
  
  self.gradInput = self.module:updateGradInput(input,gradOutput)
  return self.gradInput
end

function SpatialPyramidPooling:type(type)
  parent.type(self,type)
  self.module:type(type)
  return self
end
