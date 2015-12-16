-- for backward compatibility with saved models
local SpatialAveragePooling, parent = torch.class('inn.SpatialAveragePooling', 'nn.SpatialAveragePooling')

function SpatialAveragePooling:__init(...)
  parent.__init(self,...)
  self:ceil()
end
