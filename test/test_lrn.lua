dofile 'init.lua'
require 'ccn2'

-- Only checks output and gradinput with ccn2

module = inn.SpatialCrossResponseNormalization(5)

rmodule = nn.Sequential()
rmodule:add(nn.Transpose({1,4},{1,3},{1,2}))
rmodule:add(ccn2.SpatialCrossResponseNormalization(5))
rmodule:add(nn.Transpose({4,1},{4,2},{4,3}))
rmodule:cuda()

input = torch.randn(32,96,17,17):cuda()
gradoutput = torch.randn(#input):cuda()

output = module:forward(input)
routput = rmodule:forward(input)

gradinput = module:backward(input, gradoutput)
rgradinput = rmodule:backward(input, gradoutput)

print('Difference on input:', (output - routput):abs():mean())
print('Difference on output:', (gradinput - rgradinput):abs():mean())


