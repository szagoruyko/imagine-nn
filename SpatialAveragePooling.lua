-- for backward compatibility with saved models
local SpatialAveragePooling, parent = torch.class('inn.SpatialAveragePooling', 'nn.SpatialAveragePooling')

function SpatialAveragePooling:__init(...)
  parent.__init(self,...)
  self:ceil()
end

function SpatialAveragePooling:updateOutput(input)
   self.ceil_mode = true
   return parent.updateOutput(self, input)
end
