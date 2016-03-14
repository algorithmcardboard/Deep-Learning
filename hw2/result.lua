require 'torch'
require 'image'
require 'cunn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')
model_path = "logs/kmeans/model.net0.735"
local c = require 'trepl.colorize'
f = io.open('predictions.csv', 'w')
f:write("Id,Prediction" .. "\n")

do
  local CentroidFeatures,parent = torch.class('nn.CentroidFeatures', 'nn.SpatialConvolution')

  function CentroidFeatures:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    parent.__init(self,nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    self.train = true
    --self.weight:copy(centroids)
  end

  function CentroidFeatures:updateOutput(input)
    self.output:set(input)
    return self.output
  end

  function CentroidFeatures:updateGradInput(input, gradOutput)
    return self.gradInput
  end

  function CentroidFeatures:accGradParameters(input, gradOutput, scale)
  end
end

function parseData(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
        t[idx]:copy(this_d[j])
        l[idx] = i
        idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

--
--function parseData(d, numSamples, numChannels, height, width)
 -- local t = torch.ByteTensor(numSamples, numChannels, height, width)
  --local idx = 1
  --for i = 1, #d do
   -- t[idx]:copy(d[idx])
    --idx = idx + 1
    --if idx % 1000 == 0 then
     -- print("processed "..idx)
    --end
  --end
  --assert(idx == numSamples+1)
  --return t
--end

function normalize(unsupData)
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,unsupData:size() do
     xlua.progress(i, unsupData:size())
     -- rgb -> yuv
     local rgb = unsupData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     unsupData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = unsupData.data:select(2,2):mean()
  local std_u = unsupData.data:select(2,2):std()
  unsupData.data:select(2,2):add(-mean_u)
  unsupData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = unsupData.data:select(2,3):mean()
  local std_v = unsupData.data:select(2,3):std()
  unsupData.data:select(2,3):add(-mean_v)
  unsupData.data:select(2,3):div(std_v)
  unsupData.mean_u = mean_u
  unsupData.std_u = std_u
  unsupData.mean_v = mean_v
  unsupData.std_v = std_v
  return unsupData
end

tar = 'https://inclass.kaggle.com/c/assignment-2-stl-10/download/test.t7b'
data_path = 'stl-10'
test_file = paths.concat(data_path, 'test.t7b')
print("test file is "..test_file)

if not paths.filep(test_file) then
   os.execute('wget ' .. tar)
end

raw_test = torch.load('stl-10/test.t7b')

testData = {
   data = torch.Tensor(),
   labels = torch.Tensor(),
   size = function() return 8000 end
}

testData.data, testData.labels = parseData(raw_test.data, 8000, 3, 96, 96)

testData.data = testData.data:float()
testData.labels = testData.labels:float()

print("Test data labels")
print(testData.labels:size())

print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

testData = normalize(testData)

model = torch.load(model_path)
model:evaluate()

print(c.blue '==>'.." valing")
local bs = 25
error_count = 0
for i=1,testData.data:size(1),bs do
  local outputs = model:forward(testData.data:narrow(1,i,bs):cuda())
  local max_vals, predictions = torch.max(outputs, 2)

  for j=1,bs do
    local cIndex = (i-1)+j
    if(predictions[j][1] ~= testData.labels[cIndex]) then
      error_count = error_count + 1
    end

    print(tostring(cIndex).. "," .. tostring(predictions[j][1]))
    f:write(tostring(cIndex).. "," .. tostring(predictions[j][1]) .. '\n')
  end
end

f.close()

print("Accuracy is " .. 1-(error_count/8000))
