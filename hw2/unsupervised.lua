require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

function parseData(d, numSamples, numChannels, height, width)
  local t = torch.ByteTensor(numSamples, numChannels, height, width)
  local idx = 1
  for i = 1, #d do
    t[idx]:copy(d[idx])
    idx = idx + 1
    if idx % 1000 == 0 then
      print("processed "..idx)
    end
  end
  assert(idx == numSamples+1)
  return t
end

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

raw_extra = torch.load('stl-10/extra.t7b')

unsupData = {
   data = torch.Tensor(),
   size = function() return 100000 end
}
unsupData.data = parseData(raw_extra.data[1], 100000, 3, 96, 96)

unsupData.data = unsupData.data:float()

print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()

unsupData = normalize(unsupData)

torch.save('extra.t7',unsupData)
