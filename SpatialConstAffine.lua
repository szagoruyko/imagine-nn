local SpatialConstAffine, parent = torch.class('inn.SpatialConstAffine', 'nn.Module')

function SpatialConstAffine:__init(nOutputPlane, inplace)
  parent.__init(self)
  self.a = torch.Tensor(1,nOutputPlane,1,1)
  self.b = torch.Tensor(1,nOutputPlane,1,1)
  if self.inplace == nil then
     self.inplace = false
  else
     self.inplace = inplace
  end
  self:reset()
end

function SpatialConstAffine:reset()
  self.a:fill(1)
  self.b:zero()
end

function SpatialConstAffine:updateOutput(input)
  if self.inplace then
    self.output:set(input)
  else
    self.output:resizeAs(input):copy(input)
  end
  self.output:cmul(self.a:expandAs(input))
  self.output:add(self.b:expandAs(input))
  return self.output
end

function SpatialConstAffine:updateGradInput(input, gradOutput)
  if self.inplace then
    self.gradInput:set(gradOutput)
  else
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  end
  self.gradInput:cmul(self.a:expandAs(gradOutput))
  return self.gradInput
end
