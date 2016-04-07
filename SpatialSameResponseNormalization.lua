local SpatialSameResponseNormalization, parent = torch.class('inn.SpatialSameResponseNormalization', 'nn.Module')

function SpatialSameResponseNormalization:__init(size, alpha, beta)
  parent.__init(self)
  
  self.size  = size  or 3
  self.alpha = alpha or 5e-5
  self.beta  = beta  or 0.75
  
  local pad = math.floor(self.size/2)
  
  local numerator = nn.Identity()
  local denominator = nn.Sequential()
    :add(nn.Square())
    :add(nn.SpatialAveragePooling(self.size,self.size,1,1,pad,pad))
    :add(nn.MulConstant(self.alpha,true))
    :add(nn.AddConstant(1,true))
    :add(nn.Power(self.beta))

  local divide = nn.ConcatTable()
    :add(numerator)
    :add(denominator)
  
  self._modules = nn.Sequential()
  self._modules:add(divide)
  self._modules:add(nn.CDivTable())
end

function SpatialSameResponseNormalization:updateOutput(input)
  self.output = self._modules:updateOutput(input)
  return self.output
end

function SpatialSameResponseNormalization:updateGradInput(input, gradOutput)
  self.gradInput = self._modules:updateGradInput(input, gradOutput)
  return self.gradInput
end
