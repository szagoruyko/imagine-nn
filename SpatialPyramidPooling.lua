local SpatialPyramidPooling, parent = torch.class('inn.SpatialPyramidPooling', 'nn.Module')

-- allows nn.SpatialPyramidPooling({4,4},{3,3}) or {{4,4},{3,3}}
function SpatialPyramidPooling:__init(...)
   parent.__init(self)
   local pyr = {...}
   self.pyr = torch.type(pyr[1][1]) == 'table' and pyr[1] or pyr
   self._modules = nn.Concat(1)
   for k,v in ipairs(self.pyr) do
      self._modules:add(nn.Sequential()
         :add(nn.SpatialAdaptiveMaxPooling(v[1], v[2]))
         :add(nn.View(-1):setNumInputDims(3))
         :add(nn.Contiguous())
      )
   end
end

function SpatialPyramidPooling:updateOutput(input)
   assert(input:dim() == 4 or input:dim() == 3, 'unsupported dimensionality')
   self._modules.dimension = input:dim() - 2
   self.output = self._modules:updateOutput(input)
   return self.output
end

function SpatialPyramidPooling:updateGradInput(input, gradOutput)
   assert(input:dim() == 4 or input:dim() == 3, 'unsupported dimensionality')
   self._modules.dimension = input:dim() - 2
   self.gradInput = self._modules:updateGradInput(input, gradOutput)
   return self.gradInput
end

