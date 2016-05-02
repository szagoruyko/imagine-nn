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

  local roi_start_w = torch.round(pred_ctr_x - 0.5*(pred_w-1)) ; 
  local roi_start_h = torch.round(pred_ctr_y - 0.5*(pred_h-1)) ; 
  local roi_end_w =   torch.round(pred_ctr_x + 0.5*(pred_w-1)) ; 
  local roi_end_h =   torch.round(pred_ctr_y + 0.5*(pred_h-1)) ; 

  return torch.cat({roi_start_w, roi_start_h, roi_end_w, roi_end_h}, 2)
end

local inn = require 'inn'
local nn = require 'nn'

local n_images = 2
local channels = 3
local height = 3
local width = 6
local H = 4
local W = 3

local sz = torch.Tensor{channels, height, width}
local input_image = torch.Tensor(n_images, sz[1], sz[2], sz[3]):copy(torch.linspace(1, n_images * sz[1] * sz[2] * sz[3], n_images * sz[1] * sz[2] * sz[3]):reshape(n_images, sz[1], sz[2], sz[3]))

print(input_image)

local n_rois = 1
--local n_rois = 2
local rois=torch.Tensor(n_rois,5)
for i=1,n_rois do
  idx=torch.randperm(n_images)[1]
  y=torch.randperm(sz[3])[{{1,2}}]:sort()
  x=torch.randperm(sz[2])[{{1,2}}]:sort()
  rois[{i,{}}] = torch.Tensor({idx,x[1],y[1],x[2],y[2]})
  --rois[{i,{}}] = torch.Tensor({idx,1,1,sz[3],sz[2]})
  --rois[{i,{}}] = torch.Tensor({idx,1,1,H/2,W/2})
  --rois[{i,{}}] = torch.Tensor({idx,1,1,1,1})
end
--rois[{1,{}}] = torch.Tensor({1,2,2,3,5})
--rois[{2,{}}] = torch.Tensor({1,1,5,3,6})
--rois[{1,{}}] = torch.Tensor({1,1,5,3,6})

print(rois)

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
delta_rois[{{}, {2,5}}] = torch.rand(n_rois, 4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.7887, 0.4103, 0.7086, 0.7714}:reshape(1,4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.4694, 0.1311, 0.8265, 0.1495, 0.9336, 0.4434, 0.5211, 0.1230}:reshape(2,4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.4694, 0.1311, 0.8265, 0.1495}:reshape(1,4)
--delta_rois[{{}, {2,5}}] = torch.Tensor{0.9336, 0.4434, 0.5211, 0.1230}:reshape(1,4)
print(delta_rois)
print(delta_rois_to_rois(rois[{{}, {2,5}}], delta_rois[{{}, {2,5}}]))

local output = model:forward({input_image:cuda(), rois:cuda(), delta_rois:cuda()})
local output = model:forward({input_image:clone():fill(1):cuda(), rois:cuda(), delta_rois:cuda()})
print(output)
print(output:sum())

print('-------------------------')
local gradOutput = torch.ones(n_rois, channels, H, W):cuda() --torch.rand(n_rois, channels, H, W):cuda() --torch.Tensor(n_rois, channels, 3, 3):fill(1)
local gradInput = model:backward({input_image:cuda(), rois:cuda(), delta_rois:cuda()}, gradOutput)
--local gradInput = model:backward({input_image:cuda(), rois:cuda(), delta_rois:cuda()}, output)
print(gradInput[1])
print(gradInput[1]:sum())
print(gradInput[2])
print(gradInput[3])
print(gradInput[3]:sum())

--[[
local jac = nn.Jacobian
local err = jac.testJacobian(model, {input_image, rois, delta_rois}, nil, nil, 1e-3)
print(err)
local b = jac.backward(model, {input_image, rois, delta_rois})
print(b)
]]
