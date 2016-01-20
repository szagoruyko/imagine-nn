local SpatialPyramidPooling, parent = torch.class('inn.SpatialPyramidPooling', 'nn.Concat')

-- allows nn.SpatialPyramidPooling({4,4},{3,3}) or {{4,4},{3,3}}
function SpatialPyramidPooling:__init(...)
   parent.__init(self, 1)
   local pyr = {...}
   self.pyr = torch.type(pyr[1][1]) == 'table' and pyr[1] or pyr
   for k,v in ipairs(self.pyr) do
      parent.add(self, nn.Sequential()
         :add(nn.SpatialAdaptiveMaxPooling(v[1], v[2]))
         :add(nn.View(-1):setNumInputDims(3))
         :add(nn.Contiguous())
      )
   end
end

function SpatialPyramidPooling:updateOutput(input)
   assert(input:dim() == 4 or input:dim() == 3, 'unsupported dimensionality')
   self.dimension = input:dim() - 2
   return parent.updateOutput(self, input)
end

