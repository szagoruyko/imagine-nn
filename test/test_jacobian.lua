require 'inn'

local mytester = torch.Tester()
local jac

local precision = 1e-3

local inntest = torch.TestSuite()

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

--[[
function inntest.SpatialStochasticPooling()
   local from = math.random(1,5)
   local ki = math.random(1,4)
   local kj = math.random(1,4)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(4,5)
   local outj = math.random(4,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local input = torch.rand(from,ini,inj):cuda()
   local module = inn.SpatialStochasticPooling(ki,kj,si,sj):cuda()
   module.updateOutput = function(...)
      cutorch.manualSeed(11)
      return inn.SpatialStochasticPooling.updateOutput(...)
   end

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,5)
   input = torch.rand(nbatch,from,ini,inj):cuda()

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

function inntest.SpatialPyramidPooling()
   local from = math.random(1,5)
   local ki = math.random(1,4)
   local kj = math.random(1,4)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(4,5)
   local outj = math.random(4,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = inn.SpatialPyramidPooling({4,4},{3,3})
   local input = torch.rand(from,ini,inj)

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,5)
   input = torch.rand(nbatch,from,ini,inj)
   module = inn.SpatialPyramidPooling({4,4},{3,3})

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

function inntest.SpatialSameResponseNormalization()
    local from = 16
    local inj = 2
    local ini = inj

    local module = inn.SpatialSameResponseNormalization(3, 5e-5, 0.75)
    local input = torch.randn(from,inj,ini)

    local err = jac.testJacobian(module,input,nil,nil,1e-3)
    mytester:assertlt(err, precision, 'error on state ')

    -- batch
    local bs = 32
    local input = torch.randn(bs,from,inj,ini)
    local module = inn.SpatialSameResponseNormalization(3, 5e-5, 0.75)

    local err = jac.testJacobian(module, input, nil, nil, 1e-3)
    mytester:assertlt(err, precision, 'error on state (Batch) ')
end
]]
function randROI(sz, n)
  assert(sz:size()==4, "need 4d size")
  local roi=torch.Tensor(n,5)
  for i=1,n do
    idx=torch.randperm(sz[1])[1]
    y=torch.randperm(sz[3])[{{1,2}}]:sort()
    x=torch.randperm(sz[4])[{{1,2}}]:sort()
    roi[{i,{}}] = torch.Tensor({idx,x[1],y[1],x[2],y[2]})
  end
  return roi
end

function testJacobianWithRandomROI(cls, v2)
  --pooling grid size
  local w=4; 
  local h=4;
  --input size 
  local W=w*2;
  local H=h*2;

  local batchSize = 3
  local numRoi = batchSize
  local numRepeat = 3

  torch.manualSeed(0)
  for i=1,numRepeat do
    local input = torch.rand(batchSize, 1, H, W);
    local roi = randROI(input:size(), numRoi)
    local module = cls.new(w, h, 1, roi)
    module.v2 = v2
    local err = jac.testJacobian(module, input, nil, nil, 1e-3)
    mytester:assertlt(err, precision, 'error on ROIPooling '..(v2 and 'v2' or 'v1'))
  end
end

function inntest.ROIPooling()
  local FixedROIPooling, parent = torch.class('FixedROIPooling', 'inn.ROIPooling')
  function FixedROIPooling:__init(W, H, s, roi)
    self.roi = roi 
    parent.__init(self, W, H, s)
    self:cuda()
  end

  function FixedROIPooling:updateOutput(input)
    return parent.updateOutput(self,{input:cuda(), self.roi})
  end
  function FixedROIPooling:updateGradInput(input, gradOutput)
    return parent.updateGradInput(self,{input:cuda(), self.roi}, gradOutput)[1]
  end

  testJacobianWithRandomROI(FixedROIPooling, true)
end

function testJacobianWithRandomROIForROIWarpingData(cls)
  --pooling grid size
  local w=4; 
  local h=4;
  --input size 
  local W=w*2;
  local H=h*2;

  local batchSize = 3
  local numRoi = batchSize
  local numRepeat = 3

  torch.manualSeed(0)
  for i=1,numRepeat do
    local input = torch.rand(batchSize, 1, H, W);
    local roi = randROI(input:size(), numRoi)
    local delta_roi = roi:clone()
    delta_roi[{{}, {2, 5}}] = torch.rand(numRoi, 4) 
    local module = cls.new(w, h, 1, roi, delta_roi)

    local orig = input:clone() 
    local err = jac.testJacobian(module, input, nil, nil, 1e-3)
    mytester:assertlt(err, precision, 'error on ROIWarping ')
  end
end

function inntest.ROIWarpingData()
  local FixedROIWarping, parent = torch.class('FixedROIWarping', 'inn.ROIWarping')
  function FixedROIWarping:__init(W, H, s, roi, delta_roi)
    self.roi = roi
    self.delta_roi = delta_roi
    parent.__init(self, W, H, s)
    self:cuda()
  end

  function FixedROIWarping:updateOutput(input)
    return parent.updateOutput(self,{input:cuda(), self.roi, self.delta_roi})
  end
  function FixedROIWarping:updateGradInput(input, gradOutput)
    return parent.updateGradInput(self,{input:cuda(), self.roi, self.delta_roi}, gradOutput)[1]
  end

  testJacobianWithRandomROIForROIWarpingData(FixedROIWarping)
end
--[[
----------------------------------------------------------------------
function testJacobianWithRandomROIForROIWarpingDeltaROI(cls)
  --pooling grid size
  local w=4;
  local h=4;
  --img size
  local W=w*2;
  local H=h*2;
  

  local batchSize = 3
  local numRoi = batchSize
  local numRepeat = 3

  torch.manualSeed(0)
  for i=1,numRepeat do
    local img = torch.rand(batchSize, 3, H, W);
    --local roi = torch.Tensor{1, 1, 1, W, H}:reshape(1, 5)
    local roi = randROI(img:size(), numRoi)
    local input = torch.rand(numRoi, 4)
    local module = cls.new(w, h, 1, roi, img)

    print('---0000000000000000000000000000')
    --print(img)
    print(roi)
    print(input)
    print(delta_rois_to_rois(roi[{{}, {2,5}}], input))

    local perturbation = 1e-3
    local jac_fprop = jac.forward(module, input, input, 1e-3) 
    --module:forward(input)
    local jac_bprop = jac.backward(module, input)
 
    --print('---1111111111111111111111111111')
    print(jac_fprop)
    print('---2222222222222222222222222222')
    print(jac_bprop)   
   
    local err = jac.testJacobian(module, input, -1, 1, 1e-3)
    mytester:assertlt(err, precision, 'error on ROIWarping ')
  end
end

function inntest.ROIWarpingDeltaROI()
  local FixedROIWarpingDeltaROI, parent = torch.class('FixedROIWarpingDeltaROI', 'inn.ROIWarping')
  function FixedROIWarpingDeltaROI:__init(W, H, s, roi, img)
    self.img = img
    self.roi = roi
    self.delta_roi = self.roi:clone()
    parent.__init(self, W, H, s)
    self:cuda()
  end

  function FixedROIWarpingDeltaROI:updateOutput(input)
    self.delta_roi[{{},{2,5}}] = input:typeAs(self.delta_roi)
    return parent.updateOutput(self,{self.img:cuda(), self.roi:cuda(), self.delta_roi:cuda()})
  end
  function FixedROIWarpingDeltaROI:updateGradInput(input, gradOutput)
    self.delta_roi[{{},{2,5}}] = input:typeAs(self.delta_roi) 
    return parent.updateGradInput(self,{self.img:cuda(), self.roi:cuda(), self.delta_roi:cuda()}, gradOutput)[3][{{}, {2, 5}}]
  end

  testJacobianWithRandomROIForROIWarpingDeltaROI(FixedROIWarpingDeltaROI)
end
]]
----------------------------------------------------------------------
function testJacobianWithRandomROIForROIWarpingGridGenerator(cls)
  --pooling grid size
  local w=4;
  local h=4;
  --img size
  local W=w*2;
  local H=h*2;

  local batchSize = 3
  local numRoi = batchSize
  local numRepeat = 3

  torch.manualSeed(0)
  for i=1,numRepeat do
    local img = torch.rand(batchSize, 3, H, W);
    --local roi = torch.Tensor{1, 1, 1, W, H}:reshape(1, 5)
    local roi = randROI(img:size(), numRoi)
    local input = torch.rand(numRoi, 4)
    local module = cls.new(h, w, roi)

    local perturbation = 1e-3
    local jac_fprop = jac.forward(module, input, input, 1e-3)
    local jac_bprop = jac.backward(module, input)
 
    local err = jac.testJacobian(module, input, -1, 1, 1e-3)
    mytester:assertlt(err, precision, 'error on ROIWarping ')
  end
end

function inntest.ROIWarpingGridGeneratorGridCtrs()
  local FixedROIWarpingGridGeneratorGridCtrs, parent = torch.class('FixedROIWarpingGridGeneratorGridCtrs', 'inn.ROIWarpingGridGenerator')
  function FixedROIWarpingGridGeneratorGridCtrs:__init(H, W, roi)
    parent.__init(self, H, W)
    self.roi = roi
    self.delta_roi = self.roi:clone()
    self.grad_bin_sizes = torch.zeros(roi:size(1), 2)
    self:cuda()
  end

  function FixedROIWarpingGridGeneratorGridCtrs:updateOutput(input)
    self.delta_roi[{{},{2,5}}] = input:typeAs(self.delta_roi)
    local tmp = parent.updateOutput(self, {self.roi:cuda(), self.delta_roi:cuda()})
    self.output = self.output or input:cuda().new()
    self.output:resizeAs(tmp[1]):copy(tmp[1])
    return self.output
  end
  function FixedROIWarpingGridGeneratorGridCtrs:updateGradInput(input, gradOutput)
    self.delta_roi[{{},{2,5}}] = input:typeAs(self.delta_roi)
    self.gradInput = parent.updateGradInput(self,{self.roi:cuda(), self.delta_roi:cuda()}, {gradOutput, self.grad_bin_sizes:cuda()})
    return self.gradInput[2][{{}, {2, 5}}]
  end

  testJacobianWithRandomROIForROIWarpingGridGenerator(FixedROIWarpingGridGeneratorGridCtrs)
end

function inntest.ROIWarpingGridGeneratorBinSizes()
  local FixedROIWarpingGridGeneratorBinSizes, parent = torch.class('FixedROIWarpingGridGeneratorBinSizes', 'inn.ROIWarpingGridGenerator')
  function FixedROIWarpingGridGeneratorBinSizes:__init(H, W, roi)
    parent.__init(self, H, W)
    self.roi = roi
    self.delta_roi = self.roi:clone()
    self.grad_grid_ctrs = torch.zeros(roi:size(1), H, W, 2)
    self:cuda()
  end

  function FixedROIWarpingGridGeneratorBinSizes:updateOutput(input)
    self.delta_roi[{{},{2,5}}] = input:typeAs(self.delta_roi)
    local tmp = parent.updateOutput(self, {self.roi:cuda(), self.delta_roi:cuda()})
    self.output = self.output or input:cuda().new()
    self.output:resizeAs(tmp[2]):copy(tmp[2])
    return self.output
  end
  function FixedROIWarpingGridGeneratorBinSizes:updateGradInput(input, gradOutput)
    self.delta_roi[{{},{2,5}}] = input:typeAs(self.delta_roi)
    self.gradInput = parent.updateGradInput(self,{self.roi:cuda(), self.delta_roi:cuda()}, {self.grad_grid_ctrs:cuda(), gradOutput})
    return self.gradInput[2][{{}, {2, 5}}]
  end

  testJacobianWithRandomROIForROIWarpingGridGenerator(FixedROIWarpingGridGeneratorBinSizes)
end

function testJacobianWithRandomROIForROIWarpingBilinearSampleData(cls)
  --pooling grid size
  local w=4;
  local h=4;
  --input size
  local W=w*2;
  local H=h*2;

  local batchSize = 3
  local numRoi = batchSize
  local numRepeat = 3

  torch.manualSeed(0)
  for i=1,numRepeat do
    local input = torch.rand(batchSize, 1, H, W);
    local roi = randROI(input:size(), numRoi)
    local delta_roi = roi:clone()
    delta_roi[{{}, {2, 5}}] = torch.rand(numRoi, 4)
    local pred_rois = delta_rois_to_rois(roi[{{}, {2, 5}}], delta_roi[{{}, {2, 5}}])

    local preprocess = inn.ROIWarpingGridGenerator(h, w); preprocess:cuda()
    local output = preprocess:forward({roi:cuda(), delta_roi:cuda()})
    local grid_ctrs = output[1]:clone()
    local bin_sizes = output[2]:clone()
    local roi_batch_inds = output[3]:clone() 

    local module = cls.new(h, w, grid_ctrs, bin_sizes, roi_batch_inds)
    local err = jac.testJacobian(module, input, nil, nil, 1e-3)
    mytester:assertlt(err, precision, 'error on ROIWarpingBilinearSampleData ')
  end
end

function inntest.ROIWarpingBlinearSampleData()
  local FixedROIWarpingBilinearSampleData, parent = torch.class('FixedROIWarpingBilinearSampleData', 'inn.ROIWarpingBilinearSample')
  function FixedROIWarpingBilinearSampleData:__init(H, W, grid_ctrs, bin_sizes, roi_batch_inds)
    self.grid_ctrs = grid_ctrs:clone() 
    self.bin_sizes = bin_sizes:clone() 
    self.roi_batch_inds = roi_batch_inds:clone()
    parent.__init(self, H, W)
    self:cuda()
  end

  function FixedROIWarpingBilinearSampleData:updateOutput(input)
    return parent.updateOutput(self, {input:cuda(), self.grid_ctrs:cuda(), self.bin_sizes:cuda(), self.roi_batch_inds:cuda()})
  end
  function FixedROIWarpingBilinearSampleData:updateGradInput(input, gradOutput)
    return parent.updateGradInput(self, {input:cuda(), self.grid_ctrs:cuda(), self.bin_sizes:cuda(), self.roi_batch_inds:cuda()}, gradOutput:cuda())[1]
  end

  testJacobianWithRandomROIForROIWarpingBilinearSampleData(FixedROIWarpingBilinearSampleData)
end

function testJacobianWithRandomROIForROIWarpingBilinearSampleGridCtrs(cls)
  --pooling grid size
  local w=4;
  local h=4;
  --input size
  local W=w*2;
  local H=h*2;

  local batchSize = 3
  local numRoi = batchSize
  local numRepeat = 3

  torch.manualSeed(0)
  for i=1,numRepeat do
    local img = torch.rand(batchSize, 1, H, W);
    local roi = randROI(img:size(), numRoi)
    local delta_roi = roi:clone()
    delta_roi[{{}, {2, 5}}] = torch.rand(numRoi, 4)
    local pred_rois = delta_rois_to_rois(roi[{{}, {2, 5}}], delta_roi[{{}, {2, 5}}])

    local preprocess = inn.ROIWarpingGridGenerator(h, w); preprocess:cuda()
    local output = preprocess:forward({roi:cuda(), delta_roi:cuda()})
    local input = output[1]:clone() -- local grid_ctrs = output[1]:clone()
    local bin_sizes = output[2]:clone()
    local roi_batch_inds = output[3]:clone() 

    --print(input:select(4,1))
    --print(input:select(4,2))
    print(roi)
    print(delta_roi)
    print(pred_rois)
    --print(grid_ctrs:select(4, 1))
    --print(grid_ctrs:select(4, 2))
    --print(bin_sizes)
    --print(roi_batch_inds)

    local module = cls.new(h, w, img, bin_sizes, roi_batch_inds)

    local err = jac.testJacobian(module, input, nil, nil, 1e-3)
    -- for debug
    --if err > precision then 
    --  local jac_fprop = jac.forward(module, input, nil, 1e-3)
    --  local jac_bprop = jac.backward(module, input)
    --  print(jac_fprop)
    --  print(jac_bprop)
    --  local err = jac_fprop-jac_bprop
    --  local val, index = torch.max(err:view(-1):abs(), 1)
    --  print(val)
    --  print(index)
    --  print(input:numel())
    --  print(input:size())
    --  print(pred_rois)
    --  --print(input)
    --  print(input:select(4,1))
    --  print(grid_ctrs:select(4,1))
    --  --print(input:select(4,2))
    --end
    -- til here 
    mytester:assertlt(err, precision, 'error on ROIWarpingBilinearSample ')
  end
end

function inntest.ROIWarpingBlinearSampleGridCtrs()
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

  testJacobianWithRandomROIForROIWarpingBilinearSampleGridCtrs(FixedROIWarpingBilinearSampleGridCtrs)
end
--[[
function testJacobianWithRandomROIForROIWarpingBilinearSampleBinSizes(cls)

  --pooling grid size
  local w=4;
  local h=4;
  --input size
  local W=w*2;
  local H=h*2;

  local batchSize = 3
  local numRoi = batchSize
  local numRepeat = 3

  torch.manualSeed(0)
  for i=1,numRepeat do
    local img = torch.rand(batchSize, 1, H, W);
    local roi = randROI(img:size(), numRoi)
    local delta_roi = roi:clone()
    delta_roi[{{}, {2, 5}}] = torch.rand(numRoi, 4)
    local pred_rois = delta_rois_to_rois(roi[{{}, {2, 5}}], delta_roi[{{}, {2, 5}}])

    local preprocess = inn.ROIWarpingGridGenerator(h, w); preprocess:cuda()
    local output = preprocess:forward({roi:cuda(), delta_roi:cuda()})
    local grid_ctrs = output[1]:clone()
    local bin_sizes = output[2]:clone()
    local roi_batch_inds = output[3]:clone()

    local input = bin_sizes:clone()

    --print(input:select(4,1))
    --print(input:select(4,2))
    --print(roi)
    --print(delta_roi)
    --print(pred_rois)
    --print(grid_ctrs:select(4, 1))
    --print(grid_ctrs:select(4, 2))
    --print(bin_sizes)
    --print(roi_batch_inds)

    local module = cls.new(h, w, img, grid_ctrs, roi_batch_inds)

    local err = jac.testJacobian(module, input, nil, nil, 1e-3)
    -- for debug
    if err > precision then
      local jac_fprop = jac.forward(module, input, nil, 1e-3)
      local jac_bprop = jac.backward(module, input)
      print(jac_fprop)
      print(jac_bprop)
      local err = jac_fprop-jac_bprop
      local val, index = torch.max(err:view(-1):abs(), 1)
      print(val)
      print(index)
      print(input:numel())
      print(input:size())
      print(pred_rois)
      print(input)
      print(bin_sizes)
    end
    -- til here
    mytester:assertlt(err, precision, 'error on ROIWarpingBilinearSample ')
  end
end

function inntest.ROIWarpingBlinearSampleBinSizes()
  local FixedROIWarpingBilinearSampleBinSizes, parent = torch.class('FixedROIWarpingBilinearSampleBinSizes', 'inn.ROIWarpingBilinearSample')
  function FixedROIWarpingBilinearSampleBinSizes:__init(H, W, img, grid_ctrs, roi_batch_inds)
    self.img = img:clone()
    self.grid_ctrs = grid_ctrs:clone()
    self.roi_batch_inds = roi_batch_inds:clone()
    parent.__init(self, H, W)
    self:cuda()
  end

  function FixedROIWarpingBilinearSampleBinSizes:updateOutput(input)
    return parent.updateOutput(self, {self.img:cuda(), self.grid_ctrs:cuda(), input:cuda(), self.roi_batch_inds:cuda()})
  end
  function FixedROIWarpingBilinearSampleBinSizes:updateGradInput(input, gradOutput)
    return parent.updateGradInput(self, {self.img:cuda(), self.grid_ctrs:cuda(), input:cuda(), self.roi_batch_inds:cuda()}, gradOutput:cuda())[3]
  end

  testJacobianWithRandomROIForROIWarpingBilinearSampleBinSizes(FixedROIWarpingBilinearSampleBinSizes)
end
]]

jac = nn.Jacobian
mytester:add(inntest)
mytester:run()
