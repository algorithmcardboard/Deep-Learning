require 'nn'

local vgg = nn.Sequential()
do
  local CentroidFeatures,parent = torch.class('nn.CentroidFeatures', 'nn.SpatialConvolution')

  function CentroidFeatures:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    parent.__init(self,nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    self.train = true
    --self.weight:copy(centroids)
  end

  function CentroidFeatures:updateOutput(input)
    -- print(self.weight:size())
    self.output:set(input)
    return self.output
  end

  function CentroidFeatures:updateGradInput(input, gradOutput)
    return self.gradInput
  end

  function CentroidFeatures:accGradParameters(input, gradOutput, scale)
  end
end

centroids=torch.load('/home/ubuntu/ajr619/Deep-Learning/hw2/kernels.t7'):float()
centroids=centroids:reshape(512, 3, 13, 13):float()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end


-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

local first_layer = nn.CentroidFeatures(3,512,13,13, nil, nil, nil, nil)
first_layer.weight = centroids --:reshape(512, 3, 13, 13):float()
vgg:add(first_layer)


-- ConvBNReLU(3,64)
nInputPlane = 3
nOutputPlane = 64
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
vgg:add(MaxPooling(4,4,4,4):ceil())

-- ConvBNReLU(64,128)
nInputPlane = 64
nOutputPlane = 128
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
vgg:add(MaxPooling(3,3,3,3):ceil())

--ConvBNReLU(128,256)
nInputPlane = 128
nOutputPlane = 256
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
--ConvBNReLU(256,256)
nInputPlane = 256
nOutputPlane = 256
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))

vgg:add(MaxPooling(2,2,2,2):ceil())

-- ConvBNReLU(256,256)
nInputPlane = 256
nOutputPlane = 256
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
--ConvBNReLU(256,256)
nInputPlane = 256
nOutputPlane = 256
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
--ConvBNReLU(256,256)
nInputPlane = 256
nOutputPlane = 256
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))

vgg:add(MaxPooling(2,2,2,2):ceil())

vgg:add(nn.View(256*2*2))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*2*2,256))
classifier:add(nn.BatchNormalization(256))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256,10))
vgg:add(classifier)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
