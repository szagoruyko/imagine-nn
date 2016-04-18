local utils = {}

-- a script to simplify trained net by incorporating every Spatial/VolumetricBatchNormalization
-- to Spatial/VolumetricConvolution and BatchNormalization to Linear
local function BNtoConv(net)
  for i,v in ipairs(net.modules) do
    if v.modules then
      BNtoConv(v)
    else
      local cur = v
      local pre = net:get(i-1)
      if prev and 
        ((torch.typename(cur):find'nn.SpatialBatchNormalization' and 
          torch.typename(pre):find'nn.SpatialConvolution') or
         (torch.typename(cur):find'nn.BatchNormalization' and
          torch.typename(pre):find'nn.Linear') or
         (torch.typename(cur):find'nn.VolumetricBatchNormalization' and 
          torch.typename(pre):find'nn.VolumetricConvolution')) then
        local conv = pre
        local bn = v
        net:remove(i)
        local no = conv.nOutputPlane
        local conv_w = conv.weight:view(no,-1)
        cutorch.withDevice(conv_w:getDevice(), function()
           if bn.running_var then
              bn.running_std = bn.running_var:add(bn.eps):pow(-0.5)
           end
           if not conv.bias then
              conv.bias = bn.running_mean:clone():zero()
              conv.gradBias = conv.bias:clone()
           end
          conv_w:cmul(bn.running_std:view(no,-1):expandAs(conv_w))
          conv.bias:add(-1,bn.running_mean):cmul(bn.running_std)
          if bn.affine then
            conv.bias:cmul(bn.weight):add(bn.bias)
            conv_w:cmul(bn.weight:view(no,-1):expandAs(conv_w))
          end
          if conv.resetWeightDescriptors then
             conv:resetWeightDescriptors()
             assert(conv.biasDesc)
          end
        end)
      end
    end
  end
end

local checklist = {
  'nn.SpatialBatchNormalization',
  'nn.VolumetricBatchNormalization',
  'nn.BatchNormalization',
  'cudnn.SpatialBatchNormalization',
  'cudnn.VolumetricBatchNormalization',  
  'cudnn.BatchNormalization',
}

function utils.foldBatchNorm(net)
  -- works in place!
  BNtoConv(net)
  BNtoConv(net)
  for i,v in ipairs(checklist) do
     assert(#net:findModules(v) == 0)
  end
end


function utils.testSurgery(input, f, net, ...)
   local output1 = net:forward(input):clone()
   f(net,...)
   local output2 = net:forward(input):clone()
   local err = (output1 - output2):abs():max()
   return err
end

return utils
