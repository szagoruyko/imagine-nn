-- for backward compatibility with saved models
local SpatialMaxPooling, parent = torch.class('inn.SpatialMaxPooling', 'nn.SpatialMaxPooling')

function SpatialMaxPooling:__init(...)
  parent.__init(self,...)
  self:ceil()
end

function SpatialMaxPooling:updateOutput(input)
   self.ceil_mode = true
   return parent.updateOutput(self, input)
end
