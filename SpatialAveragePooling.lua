local C = inn.C

local SpatialAveragePooling, parent = torch.class('inn.SpatialAveragePooling', 'nn.SpatialAveragePooling')


function SpatialAveragePooling:__init(kW, kH, dW, dH)
  parent.__init(self, kW, kH, dW, dH)
  self:cuda()
end


function SpatialAveragePooling:updateOutput(input)
  C['SpatialAveragePooling_updateOutput'](cutorch.getState(), input:cdata(), 
  	self.output:cdata(), self.kW, self.kH, self.dW, self.dH)
  return self.output
end

function SpatialAveragePooling:updateGradInput(input, gradOutput)
  C['SpatialAveragePooling_updateGradInput'](cutorch.getState(), input:cdata(), 
  	gradOutput:cdata(), self.gradInput:cdata(), self.kW, self.kH, self.dW, self.dH)
  return self.gradInput
end
