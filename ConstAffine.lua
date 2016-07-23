local ConstAffine, parent = torch.class('inn.ConstAffine', 'nn.Module')

function ConstAffine:__init(nOutputPlane, inplace)
   parent.__init(self)
   self.a = torch.Tensor(nOutputPlane)
   self.b = torch.Tensor(nOutputPlane)
   if self.inplace == nil then
      self.inplace = false
   else
      self.inplace = inplace
   end
   self:reset()
end

function ConstAffine:reset()
   self.a:fill(1)
   self.b:zero()
end

local function view(self, input)
   local n = self.a:numel()
   local a,b
   if input:dim() == 2 then
      a = self.a:view(1,n)
      b = self.b:view(1,n)
   elseif input:dim() == 4 then
      a = self.a:view(1,n,1,1)
      b = self.b:view(1,n,1,1)
   else
      error'Unsupported dimension'
   end
   a = a:expandAs(input)
   b = b:expandAs(input)
   return a,b
end

function ConstAffine:updateOutput(input)
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   local a,b = view(self, input)
   self.output:cmul(a):add(b)
   return self.output
end

function ConstAffine:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   local a,b = view(self, input)
   return self.gradInput:cmul(a)
end
