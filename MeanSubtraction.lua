local MeanSubtraction, parent = torch.class('inn.MeanSubtraction', 'nn.Module')

function MeanSubtraction:__init(mean)
  parent.__init(self)
  assert(mean)

  self.mean = mean
  self.output = torch.Tensor()
end

function MeanSubtraction:updateOutput(input)
  if input:nDimension() == 3 and self.mean:nDimension() ~= 3 then
    self.mean:viewAs(input)
  elseif input:nDimension() == 4 and self.mean:nDimension() ~= 4 or self.mean:size(1) ~= input:size(1) then
    local batch_size = input:size(1)
    if self.mean:nDimension() == 4 and self.mean:size(1) ~= 1 then
      self.mean = self.mean:select(1,1)
    end
    self.mean = self.mean:view(1, input:size(2), input:size(3), input:size(4))
    self.mean = self.mean:expandAs(input)
  end
  self.output:add(input,-1,self.mean)
  return self.output
end


function MeanSubtraction:updateGradInput(input, gradOutput)
  return gradOutput
end
