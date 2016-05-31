local function delta_rois_to_rois(rois, delta_rois)
  local src_w = rois[{{},3}] - rois[{{},1}] + 1;
  local src_h = rois[{{},4}] - rois[{{},2}] + 1;
  local src_ctr_x = rois[{{},1}] + 0.5*(src_w-1.0);
  local src_ctr_y = rois[{{},2}] + 0.5*(src_h-1.0);

  local dst_ctr_x = delta_rois[{{},1}]; -- dx (in fast-rcnn notation) = cx (in here)
  local dst_ctr_y = delta_rois[{{},2}]; -- dy (in fast-rcnn notation) = cy (in here)
  local dst_scl_x = delta_rois[{{},3}]; -- dw (in fast-rcnn notation) = sx (in here)
  local dst_scl_y = delta_rois[{{},4}]; -- dh (in fast-rcnn notation) = sy (in here)

  local pred_ctr_x = torch.cmul(dst_ctr_x, src_w) + src_ctr_x; 
  local pred_ctr_y = torch.cmul(dst_ctr_y, src_h) + src_ctr_y; 
  local pred_w = torch.cmul(torch.exp(dst_scl_x), src_w);            
  local pred_h = torch.cmul(torch.exp(dst_scl_y), src_h);            

  local roi_start_w = pred_ctr_x - 0.5*(pred_w-1) 
  local roi_start_h = pred_ctr_y - 0.5*(pred_h-1) 
  local roi_end_w =   pred_ctr_x + 0.5*(pred_w-1) 
  local roi_end_h =   pred_ctr_y + 0.5*(pred_h-1) 

  return torch.cat({roi_start_w, roi_start_h, roi_end_w, roi_end_h}, 2)
end

local inn = require 'inn'
local nn = require 'nn'

torch.manualSeed(3)

local n_images = 1 -- 2 
local channels = 3
local H = 4
local W = 3
local height = 3
local width = 6

local sz = torch.Tensor{channels, height, width}
local input_image = torch.Tensor(n_images, sz[1], sz[2], sz[3]):copy(torch.linspace(1, n_images * sz[1] * sz[2] * sz[3], n_images * sz[1] * sz[2] * sz[3]):reshape(n_images, sz[1], sz[2], sz[3]))

print(input_image)

local n_rois = 1 --3 --10
local rois=torch.Tensor(n_rois,5)
for i=1,n_rois do
  idx=torch.randperm(n_images)[1]
  y=torch.randperm(sz[2])[{{1,2}}]:sort()
  x=torch.randperm(sz[3])[{{1,2}}]:sort()
  rois[{i,{}}] = torch.Tensor({idx,x[1],y[1],x[2],y[2]})
  --rois[{i,{}}] = torch.Tensor({idx,1,1,sz[3],sz[2]})
  --rois[{i,{}}] = torch.Tensor({idx,1,1,H/2,W/2})
  --rois[{i,{}}] = torch.Tensor({idx,1,1,1,1})
end
--rois[{1,{}}] = torch.Tensor({1,2,2,3,5})
--rois[{2,{}}] = torch.Tensor({1,1,5,3,6})
--rois[{1,{}}] = torch.Tensor({1,1,5,3,6})
--rois[{{}, {}}] = torch.Tensor({{1,  3,  1,  6,  5},
--                               {3,  4,  4,  5,  8},
--                               {2,  1,  1,  3,  5}})
--print(rois)

local model = inn.ROIPooling(W,H)
model.v2 = false
model:cuda()

local output = model:forward({input_image:cuda(), rois:cuda()})
--print(output)

local model = inn.ROIWarping(W,H)
model:cuda()
--local output = model:forward({input_image:cuda(), rois:cuda()})
--print(output)

---------------
print('-------------------------')
local delta_rois = rois:clone()
--delta_rois[{{}, {2,5}}] = 0 
--delta_rois[{{}, {2,5}}] = torch.ones(n_rois, 4) * 0.1 
--delta_rois[{{}, {2,5}}] = 0.1 * torch.rand(n_rois, 4)
delta_rois[{{}, {2,5}}] = torch.rand(n_rois, 4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.7887, 0.4103, 0.7086, 0.7714}:reshape(1,4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.4694, 0.1311, 0.8265, 0.1495, 0.9336, 0.4434, 0.5211, 0.1230}:reshape(2,4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.4694, 0.1311, 0.8265, 0.1495}:reshape(1,4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.9336, 0.4434, 0.5211, 0.1230}:reshape(1,4)
--delta_rois[{{}, {}}] = torch.Tensor({{1.0000, 0.7253, 0.6597, 0.5013, 0.5332},
--                                     {3.0000, 0.9561, 0.2305, 0.6440, 0.3949},
--                                     {2.0000, 0.4239, 0.6188, 0.6064, 0.4749}})
--
print(rois)
print(delta_rois)
local pred_rois = delta_rois_to_rois(rois[{{}, {2,5}}], delta_rois[{{}, {2,5}}])
print(pred_rois)
print(torch.round(pred_rois))
--[[
local output = model:forward({input_image:cuda(), rois:cuda(), delta_rois:cuda()})
--local output = model:forward({input_image:clone():fill(1):cuda(), rois:cuda(), delta_rois:cuda()})
print(output)
print(output:sum())

print('-------------------------')
local gradOutput = torch.ones(n_rois, channels, H, W):cuda() --torch.rand(n_rois, channels, H, W):cuda() --torch.Tensor(n_rois, channels, 3, 3):fill(1)
local gradInput = model:backward({input_image:cuda(), rois:cuda(), delta_rois:cuda()}, gradOutput)
--local gradInput = model:backward({input_image:clone():fill(1):cuda(), rois:cuda(), delta_rois:cuda()}, gradOutput)
print(gradInput[1])
print(gradInput[1]:sum())
print(gradInput[2])
print(gradInput[3])
print(gradInput[3]:sum())
]]

--print('------------------------------------------------------------')
local model = inn.ROIWarpingGridGenerator(H, W)
model:cuda()
local output = model:forward({rois:cuda(), delta_rois:cuda()})
--print(output[1]:select(4,1))
--print(output[1]:select(4,2))
--print(output[2])
--print(output[3])
local grid_ctrs = output[1]:clone()
local bin_sizes = output[2]:clone()
local roi_batch_inds = output[3]:clone() 

local gradOutput = {torch.ones(n_rois, H, W, 2):cuda(), --torch.rand(n_rois, channels, H, W, 2):cuda()
                    torch.ones(n_rois, 2):cuda()} 
local gradInput = model:backward({rois:cuda(), delta_rois:cuda()}, gradOutput)
--print(gradInput[1])
--print(gradInput[2])

print('------------------------------------------------------------')
--local input_image = 10 * torch.rand(input_image:size()) --torch.ones(input_image:size())
local input_image = torch.ones(input_image:size())
local model = inn.ROIWarpingBilinearSample(H, W)
model:cuda()
local output = model:forward({input_image:cuda(), grid_ctrs:cuda(), bin_sizes:cuda(), roi_batch_inds:cuda()})
--print(output)
--print(output:sum())

--print('hi0000000000000000000000')
local gradOutput = torch.ones(n_rois, channels, H, W):cuda()
local gradInput = model:backward({input_image:cuda(), grid_ctrs:cuda(), bin_sizes:cuda(), roi_batch_inds:cuda()}, gradOutput)
--print(gradInput)
--print(gradInput[1])
--print(gradInput[1]:sum())
--print(gradInput[2]:select(4,1)/3)
--print(gradInput[2]:select(4,2)/3)
--print(gradInput[2]:sum())
--print(gradInput[3])
--print(gradInput[4])

print('------------------------------------------------------------')
local FixedROIWarpingBilinearSampleGridCtrs, parent = torch.class('FixedROIWarpingBilinearSampleGridCtrs', 'inn.ROIWarpingBilinearSample')
function FixedROIWarpingBilinearSampleGridCtrs:__init(H, W, img, bin_sizes, roi_batch_inds)
  self.img = img:clone()
  self.bin_sizes = bin_sizes:clone()
  self.roi_batch_inds = roi_batch_inds:clone()
  parent.__init(self, H, W)
  self:cuda()
end

function FixedROIWarpingBilinearSampleGridCtrs:updateOutput(input)
  return parent.updateOutput(self, {self.img:cuda(), input:cuda(), self.bin_sizes:cuda(), self.roi_batch_inds:cuda()})
end
function FixedROIWarpingBilinearSampleGridCtrs:updateGradInput(input, gradOutput)
  return parent.updateGradInput(self, {self.img:cuda(), input:cuda(), self.bin_sizes:cuda(), self.roi_batch_inds:cuda()}, gradOutput:cuda())[2]
end

local jac = nn.Jacobian

local module = FixedROIWarpingBilinearSampleGridCtrs.new(H, W, input_image, bin_sizes, roi_batch_inds) 
local input = grid_ctrs:clone()

local err = jac.testJacobian(module, input, nil, nil, 1e-3)

local ind = 24 










local function jacforward(module, input, param, perturbation)
   param = param or input
   -- perturbation amount
   perturbation = perturbation or 1e-6
   -- 1D view of input
   --local tst = param:storage()
   local sin = param.new(param):resize(param:nElement())--param.new(tst,1,tst:size())
   -- jacobian matrix to calculate
   local jacobian = torch.Tensor():resize(param:nElement(),module:forward(input):nElement())

   print('hi11')
   print('input: ')
   print(input)
   print('param: ')
   print(param)
   print('sin: ') 
   print(sin)

   local outa = torch.Tensor(jacobian:size(2))
   local outb = torch.Tensor(jacobian:size(2))

   for i=1,sin:nElement() do
      local orig = sin[i]
      if i == ind then 
        print(orig)
      end

      sin[i] = orig - perturbation
      outa:copy(module:forward(input))
      if i == ind then 
        print(sin[i])
      end

      sin[i] = orig + perturbation
      outb:copy(module:forward(input))
      if i == ind then 
        print(sin[i])
      end

      sin[i] = orig

      if i == ind then  
        print(outa)
        print(outb)
        print(outb-outa)
      end 

      outb:add(-1,outa):div(2*perturbation)

      if i == 5 then
        print(outb)
      end

      jacobian:select(1,i):copy(outb)
   end

   return jacobian
end


--local input = grid_ctrs:clone()
print(input)
print(grid_ctrs)
if err > 0.001 then
  print('error!!!!!!!!!!!!!!!')
  print(err)
  --local jac_fprop = jac.forward(module, input, nil, 1e-3)
  local jac_fprop = jacforward(module, input, nil, 1e-3)
  local jac_bprop = jac.backward(module, input)   -- input:numel() x output:numel()
  print(jac_fprop)
  print(jac_bprop)
  local err = jac_fprop-jac_bprop
  local val, index = torch.max(err:view(-1):abs(), 1)
  print(val)
  print(index)
  --print(input:numel()) --print(input:nElement())
  print('input img: ')
  print(input_image:size())
  print(input_image)
  print('pred_rois: ')
  print(pred_rois)
  print(bin_sizes)
  --print(input)
  --print(input:select(4,1))
  --print(input:select(4,2))
  --print(grid_ctrs:select(4,1))
  --print(grid_ctrs:select(4,2))
  --print(grid_ctrs:view(-1))
  --print(grid_ctrs:view(-1):select(1,4))

  print('test error!!!!!!!!!!!!!!!!!!!!!!!!!!')
  local tmp = input:clone() --grid_ctrs:clone()
  local tmp2 = tmp:view(-1)
  local orig = tmp2[ind]
  --local ind = 5

  tmp2[ind] = orig + 1e-3 
  local output1 = model:forward({input_image:cuda(), tmp:cuda(), bin_sizes:cuda(), roi_batch_inds:cuda()}):clone()
  print('grid_ctrs1: ')
  print(tmp:select(4,1))
  print(tmp:select(4,2))
  print('output1: ')
  print(output1)

  tmp2[ind] = orig - 1e-3 
  local output2 = model:forward({input_image:cuda(), tmp:cuda(), bin_sizes:cuda(), roi_batch_inds:cuda()})
  print('grid_ctrs2: ')
  print(tmp:select(4,1))
  print(tmp:select(4,2))
  print('output2: ')
  print(output2)
  print(torch.sqrt(torch.pow(output1:view(-1) - output2:view(-1),2)))
  print(torch.sum(torch.sqrt(torch.pow(output1:view(-1) - output2:view(-1),2))))
end
