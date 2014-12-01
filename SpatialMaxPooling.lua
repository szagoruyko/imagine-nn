local C = inn.C

local SpatialMaxPooling, parent = torch.class('inn.SpatialMaxPooling', 'nn.SpatialMaxPooling')


function SpatialMaxPooling:__init(kW, kH, dW, dH)
  parent.__init(self, kW, kH, dW, dH)
  self:cuda()
end


function SpatialMaxPooling:updateOutput(input)
  C['SpatialMaxPooling_updateOutput'](input:cdata(), self.output:cdata(), self.indices:cdata(), self.kW, self.kH, self.dW, self.dH)
  return self.output
end

function SpatialMaxPooling:updateGradInput(input, gradOutput)
  C['SpatialMaxPooling_updateGradInput'](input:cdata(), self.gradInput:cdata(), gradOutput:cdata(), self.indices:cdata(), self.kW, self.kH, self.dW, self.dH)
  return self.gradInput
end
