local C = inn.C

local SpatialCrossResponseNormalization, parent = torch.class('inn.SpatialCrossResponseNormalization', 'nn.Module')

function SpatialCrossResponseNormalization:__init(size, alpha, beta, k)
  parent.__init(self)
  
  self.size = size
  self.alpha = alpha or 0.001
  self.beta = beta or 0.75
  self.k = k or 1

  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
  self.scale = torch.Tensor()

  self:cuda()
end

function SpatialCrossResponseNormalization:updateOutput(input)
  if input:type() ~= 'torch.CudaTensor' then error('CudaTensor expected') end
  C['LRNforward'](input:cdata(), self.output:cdata(), self.scale:cdata(), self.size, self.alpha, self.beta, self.k)
  return self.output
end
