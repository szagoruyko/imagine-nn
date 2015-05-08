local C = inn.C

local SpatialMaxPooling, parent = torch.class('inn.SpatialMaxPooling', 'nn.SpatialMaxPooling')


function SpatialMaxPooling:__init(kW, kH, dW, dH)
  parent.__init(self, kW, kH, dW, dH)
  self:cuda()
end


function SpatialMaxPooling:updateOutput(input)
  assert(torch.isTypeOf(input, 'torch.CudaTensor'))
  C.SpatialMaxPooling_updateOutput(cutorch.getState(), input:cdata(), self.output:cdata(),
  	self.indices:cdata(), self.kW, self.kH, self.dW, self.dH)
  return self.output
end

function SpatialMaxPooling:updateGradInput(input, gradOutput)
  assert(torch.isTypeOf(input, 'torch.CudaTensor'))
  assert(torch.isTypeOf(gradOutput, 'torch.CudaTensor'))
  C.SpatialMaxPooling_updateGradInput(cutorch.getState(), input:cdata(), self.gradInput:cdata(),
  	gradOutput:cdata(), self.indices:cdata(), self.kW, self.kH, self.dW, self.dH)
  return self.gradInput
end
