require 'inn'

local mytester = torch.Tester()
local jac

local precision = 1e-3

local inntest = torch.TestSuite()


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
    local module = cls.new(h, w, 1, roi)
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

function inntest.ConstAffine()
   local inj_vals = {math.random(3,5), 1}  -- Also test the inj = 1 spatial case

   for ind, inj in pairs(inj_vals) do
     local module = inn.ConstAffine(inj)

     -- 2D
     local nframe = math.random(50,70)
     local input = torch.Tensor(nframe, inj,3,3)

     local err = jac.testJacobian(module,input)
     mytester:assertlt(err,precision, 'error on state ')

     -- IO
     local ferr,berr = jac.testIO(module,input)
     mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
     mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
  end  -- for ind, inj in pairs(inj_vals) do
end


jac = nn.Jacobian
mytester:add(inntest)
mytester:run()
