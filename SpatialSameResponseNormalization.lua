local SpatialSameResponseNormalization, parent = torch.class('inn.SpatialSameResponseNormalization', 'nn.Sequential')

function SpatialSameResponseNormalization:__init(size, alpha, beta)
  parent.__init(self)
  
  self.size  = size  or 3
  self.alpha = alpha or 5e-5
  self.beta  = beta  or 0.75
  
  local pad = math.floor(self.size/2)
  
  local numerator = nn.Identity()
  local denominator = nn.Sequential()
    :add(nn.Square())
    :add(nn.SpatialAveragePooling(self.size,self.size,1,1,pad,pad))
    :add(nn.MulConstant(self.alpha,true))
    :add(nn.AddConstant(1,true))
    :add(nn.Power(self.beta))

  local divide = nn.ConcatTable()
    :add(numerator)
    :add(denominator)
  
  self:add(divide)
  self:add(nn.CDivTable())
end

