require 'inn'

local mytester = torch.Tester()
local jac

local precision = 1e-3

local inntest = {}


-- disabled test because of stochastic nature
-- to do it properly testJacobian needs to reset seed before every forward
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

   local module = inn.SpatialStochasticPooling(ki,kj,si,sj):cuda()
   local input = torch.rand(from,ini,inj):cuda()

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,5)
   input = torch.rand(nbatch,from,ini,inj):cuda()
   module = inn.SpatialStochasticPooling(ki,kj,si,sj):cuda()

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err (Batch) ')
end
]]--

function inntest.SpatialAveragePooling()
   local from = math.random(1,5)
   local ki = math.random(1,4)
   local kj = math.random(1,4)
   local si = ki
   local sj = kj
   local outi = math.random(4,5)
   local outj = math.random(4,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = inn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local input = torch.rand(from,ini,inj):cuda()

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,5)
   input = torch.rand(nbatch,from,ini,inj):cuda()
   module = inn.SpatialAveragePooling(ki,kj,si,sj):cuda()

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

function inntest.SpatialCrossResponseNormalization()
    local from = 16
    local inj = 2
    local ini = inj

    local module = inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2)
    local input = torch.randn(from,inj,ini):cuda():zero()

    local err = jac.testJacobian(module,input,nil,nil,1e-3)
    mytester:assertlt(err, precision, 'error on state ')

    -- batch
    local bs = 32
    local input = torch.randn(bs,from,inj,ini):cuda():zero()
    local module = inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2)

    local err = jac.testJacobian(module, input, nil, nil, 1e-3)
    mytester:assertlt(err, precision, 'error on state (Batch) ')
end

function inntest.SpatialSameResponseNormalization()
    local from = 16
    local inj = 2
    local ini = inj

    local module = inn.SpatialSameResponseNormalization(3, 5e-5, 0.75):cuda()
    local input = torch.randn(from,inj,ini):cuda():zero()

    local err = jac.testJacobian(module,input,nil,nil,1e-3)
    mytester:assertlt(err, precision, 'error on state ')

    -- batch
    local bs = 32
    local input = torch.randn(bs,from,inj,ini):cuda():zero()
    local module = inn.SpatialSameResponseNormalization(3, 5e-5, 0.75):cuda()

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

function testJacobianWithRandomROI(cls)
    --pooling grid size
    local w=4; 
    local h=4;
    --input size 
    local W=w*2;
    local H=h*2;

    local batchSize = 3
    local numRoi = batchSize
    local numRepeat = 2

    torch.manualSeed(0)
    print('\nTesting module '.. torch.type(cls))
    for i=1,numRepeat do
        print(i)
        local input = torch.rand(batchSize, 1, H, W);
        local roi=randROI(input:size(), numRoi)
        local module=cls.new(h, w, 1, roi)
        local err=jac.testJacobian(module, input, nil, nil, 1e-3)
        print(err)
        mytester:assertlt(err, precision, 'error on RoiPooling')
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

    testJacobianWithRandomROI(FixedROIPooling)

end

nnf={}
dofile('nnf.ROIPooling.lua')
function inntest.NNFROIPooling()
    local FixedROIPooling, parent = torch.class('NNFFixedROIPooling', 'nnf.ROIPooling')
    function FixedROIPooling:__init(W, H, s, roi)
        self.roi = roi 
        parent.__init(self, W, H)
        parent.setSpatialScale(self, s)
        self:cuda()
    end

    function FixedROIPooling:updateOutput(input)
        return parent.updateOutput(self,{input:cuda(), self.roi})
    end
    function FixedROIPooling:updateGradInput(input, gradOutput)
        return parent.updateGradInput(self,{input:cuda(), self.roi}, gradOutput)[1]
    end

    testJacobianWithRandomROI(FixedROIPooling)
end

function inntest.compareNNFAndINNROIPooling()
    --pooling grid size
    local w=4; 
    local h=4;
    local scale=1;
    --input size 
    local W=w*2;
    local H=h*2;

    local batchSize = 3
    local numRoi = batchSize
    local numRepeat = 2

    torch.manualSeed(0)
    for i=1,numRepeat do
        local input = torch.rand(batchSize, 1, H, W);
        local roi=randROI(input:size(), numRoi)
        local innModule=inn.ROIPooling(w, h, scale, roi)
        local nnfModule=nnf.ROIPooling(w, h, roi); nnfModule:setSpatialScale(scale)

        innModule:cuda()
        nnfModule:cuda()
        local inputs = {input:cuda(), roi:cuda()}

        local innOutput=innModule:forward(inputs)
        local nnfOutput=nnfModule:forward(inputs)
        local err=(innOutput:view(-1)-nnfOutput:view(-1)):norm()
        mytester:assertlt(err, precision, 'error comparing two ROIPooling forward()')

        gradOutput= torch.rand(batchSize, 1, h, w):cuda()
        innGrad=innModule:backward(inputs, gradOutput)[1]
        nnfGrad=nnfModule:backward(inputs, gradOutput)[1]
        local err=(innGrad:view(-1) - nnfGrad:view(-1)):norm()
        mytester:assertlt(err, precision, 'error comparing two ROIPooling backward()')

    end

end

jac = nn.Jacobian
mytester:add(inntest)
mytester:run()
