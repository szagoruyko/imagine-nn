require 'inn'

local mytester = torch.Tester()
local jac

local precision = 1e-3

local inntest = {}


function inntest.SpatialMaxPooling()
   local from = math.random(1,5)
   local ki = math.random(1,4)
   local kj = math.random(1,4)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(4,5)
   local outj = math.random(4,5)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local module = inn.SpatialMaxPooling(ki,kj,si,sj):cuda()
   local input = torch.rand(from,ini,inj):cuda()

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err ')

   -- batch
   local nbatch = math.random(2,5)
   input = torch.rand(nbatch,from,ini,inj):cuda()
   module = inn.SpatialMaxPooling(ki,kj,si,sj):cuda()

   local err = jac.testJacobian(module, input, nil, nil, 1e-3)
   mytester:assertlt(err, precision, 'error on state (Batch) ')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err (Batch) ')
   mytester:assertlt(berr, 1e-6, torch.typename(module) .. ' - i/o backward err (Batch) ')
end

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


jac = nn.Jacobian
mytester:add(inntest)
mytester:run()
