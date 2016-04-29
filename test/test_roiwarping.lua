local inn = require 'inn'

local n_images = 1
local channels = 3
local height = 6
local width = 6
local sz = torch.Tensor{channels, height, width}
local input_image = torch.CudaTensor(n_images, sz[1], sz[2], sz[3]):copy(torch.linspace(1, n_images * sz[1] * sz[2] * sz[3], n_images * sz[1] * sz[2] * sz[3]):reshape(n_images, sz[1], sz[2], sz[3]))

print(input_image)

local n_rois = 1
local rois=torch.CudaTensor(n_rois,5)
for i=1,n_rois do
  idx=torch.randperm(n_images)[1]
  y=torch.randperm(sz[3])[{{1,2}}]:sort()
  x=torch.randperm(sz[2])[{{1,2}}]:sort()
  rois[{i,{}}] = torch.Tensor({idx,x[1],y[1],x[2],y[2]})
  rois[{i,{}}] = torch.Tensor({idx,1,1,sz[3],sz[2]})
end

print(rois)

local model = inn.ROIPooling(3,3)
model.v2 = false
model:cuda()

local output = model:forward({input_image, rois})
print(output)

local model = inn.ROIWarping(3,3)
model:cuda()
local output = model:forward({input_image, rois})
print(output)

