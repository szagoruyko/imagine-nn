local SpatialSameResponseNormalization, parent = torch.class('inn.SpatialSameResponseNormalization', 'nn.Module')

function SpatialSameResponseNormalization:__init(size, alpha, beta)
  parent.__init(self)
  
  self.size  = size  or 3
  self.alpha = alpha or 5e-5
  self.beta  = beta  or 0.75
  
  local pad = math.floor(self.size/2)
  
  local numerator = nn.Identity()
  local denominator = nn.Sequential()
  denominator:add(nn.SpatialZeroPadding(pad, pad, pad, pad))
  denominator:add(nn.Power(2))
  denominator:add(nn.SpatialAveragePooling(self.size,self.size,1,1))
  denominator:add(nn.MulConstant(self.alpha,true))
  denominator:add(nn.AddConstant(1,true))
  denominator:add(nn.Power(self.beta))

  local divide = nn.ConcatTable()
  divide:add(numerator)
  divide:add(denominator)
  
  self.modules = nn.Sequential()
  self.modules:add(divide)
  self.modules:add(nn.CDivTable())
end

function SpatialSameResponseNormalization:updateOutput(input)
  self.output = self.modules:forward(input)
  return self.output
end

function SpatialSameResponseNormalization:updateGradInput(input,gradOutput)
  self.gradInput = self.modules:backward(input,gradOutput)
  return self.gradInput
end

function SpatialSameResponseNormalization:type(type)
  parent.type(self,type)
  self.modules:type(type)
  return self
end
