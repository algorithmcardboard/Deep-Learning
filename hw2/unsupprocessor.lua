require 'nn'
require 'image'
require 'xlua'

torch.setdefaulttensortype('torch.FloatTensor')

function parseData(d, numSamples, numChannels, height, width)
  local t = torch.ByteTensor(numSamples, numChannels, height, width)
  local idx = 1
  --for i = 1, #d do
    --t[idx]:copy(d[idx])
    --idx = idx + 1
    --if idx % 1000 == 0 then
      --print("processed "..idx)
    --end
  --end
  --assert(idx == numSamples+1)
  return t, l
end

function normalize()
end

raw_extra = torch.load('stl-10/extra.t7b')
processed = parseData(raw_extra.data[1], 100000, 3, 96, 96)
