local utils = {}

-- a script to simplify trained net by incorporating every Spatial/VolumetricBatchNormalization
-- to Spatial/VolumetricConvolution and BatchNormalization to Linear
local function BNtoConv(net)
  for i,v in ipairs(net.modules) do
    if v.modules then
      BNtoConv(v)
    else
      if net:get(i-1) and 
        ((torch.typename(v):find'nn.SpatialBatchNormalization' and 
          torch.typename(net:get(i-1)):find'SpatialConvolution') or
         (torch.typename(v):find'nn.VolumetricBatchNormalization' and 
          torch.typename(net:get(i-1)):find'VolumetricConvolution')) then
        local conv = net:get(i-1)
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

local function BNToLinear(net)
  for i,v in ipairs(net.modules) do
    if v.modules then
      BNToLinear(v)
    else
      if (torch.typename(v):find'nn.BatchNormalization') and
        (torch.typename(net:get(i-1)):find'Linear') then
        local linear = net:get(i-1)
        local bn = v
        net:remove(i)
        local no = linear.weight:size(1)
        if bn.running_var then
          bn.running_std = bn.running_var:add(x.eps):pow(-0.5)
        end
        linear.weight:cmul(bn.running_std:view(no,1):expandAs(linear.weight))
        linear.bias:add(-1,bn.running_mean):cmul(bn.running_std)
        if bn.affine then
          linear.bias:cmul(bn.weight):add(bn.bias)
          linear.weight:cmul(bn.weight:view(no,1):expandAs(linear.weight))
        end
      end
    end
  end
end

function utils.foldBatchNorm(net)
  -- works in place!
  BNtoConv(net)
  BNtoConv(net)
  BNToLinear(net)
  assert(#net:findModules'nn.SpatialBatchNormalization' == 0)
  assert(#net:findModules'nn.VolumetricBatchNormalization' == 0)
  assert(#net:findModules'nn.BatchNormalization' == 0)
  assert(#net:findModules'cudnn.SpatialBatchNormalization' == 0)
  assert(#net:findModules'cudnn.VolumetricBatchNormalization' == 0)  
  assert(#net:findModules'cudnn.BatchNormalization' == 0)
end


function utils.testSurgery(input, f, net, ...)
   local output1 = net:forward(input):clone()
   f(net,...)
   local output2 = net:forward(input):clone()
   local err = (output1 - output2):abs():max()
   return err
end

return utils
