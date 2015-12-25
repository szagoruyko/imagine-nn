local SpatialPyramidPooling, parent = torch.class('inn.SpatialPyramidPooling', 'nn.Concat')

-- allows nn.SpatialPyramidPooling({4,4},{3,3}) or {{4,4},{3,3}}
function SpatialPyramidPooling:__init(...)
   parent.__init(self, 1)
   local pyr = {...}
   self.pyr = torch.type(pyr[1][1]) == 'table' and pyr[1] or pyr
   self.dimd = 1
   for k,v in ipairs(self.pyr) do
      parent.add(self, nn.Sequential()
         :add(nn.SpatialAdaptiveMaxPooling(v[1], v[2]))
         :add(nn.View(-1):setNumInputDims(3))
         :add(nn.Contiguous())
      )
   end
end

function dimensionSet(self, input)
   local dimd = 1
   if input:nDimension() == 4 then 
      dimd = dimd + 1
   end

   if self.dimd ~= dimd then
      self.dimd = dimd
      self.dimension = dimd
   end
end

function SpatialPyramidPooling:updateOutput(input)
   dimensionSet(self, input) 
   return parent.updateOutput(self, input)
end

function SpatialPyramidPooling:updateGradInput(input, gradOutput)
   dimensionSet(self, input) 
   return parent.updateGradInput(self, input, gradOutput)
end
