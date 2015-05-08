local C = inn.C

local SpatialStochasticPooling, parent = torch.class('inn.SpatialStochasticPooling', 'nn.SpatialMaxPooling')


function SpatialStochasticPooling:__init(kW, kH, dW, dH)
  parent.__init(self, kW, kH, dW, dH)
  self.train = true
  self:cuda()
end


function SpatialStochasticPooling:updateOutput(input)
  assert(torch.isTypeOf(input, 'torch.CudaTensor'))
  C.SpatialStochasticPooling_updateOutput(cutorch.getState(), input:cdata(), self.output:cdata(),
  	self.indices:cdata(), self.kW, self.kH, self.dW, self.dH, self.train)
  return self.output
end

function SpatialStochasticPooling:updateGradInput(input, gradOutput)
  assert(torch.isTypeOf(input, 'torch.CudaTensor'))
  assert(torch.isTypeOf(gradOutput, 'torch.CudaTensor'))
  C.SpatialStochasticPooling_updateGradInput(cutorch.getState(), input:cdata(), self.gradInput:cdata(),
  	gradOutput:cdata(), self.indices:cdata(), self.kW, self.kH, self.dW, self.dH)
  return self.gradInput
end
