require 'image'
require 'cunn'
require 'torch'
m = require 'manifold'

dofile 'models/kmeans.lua'

function parseDataLabel(d, numSamples, numChannels, height, width)
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

raw_train = torch.load('stl-10/train.t7b')

trainData = {
  data = torch.Tensor(),
  labels = torch.Tensor(),
  size = function() return 4000 end
}
trainData.data, trainData.labels = parseDataLabel(raw_train.data, 4000, 3, 96, 96)

perm = torch.randperm(4000)[{{1,1000}}]:long()

X = trainData.data:index(1, perm):float()
y = trainData.labels:index(1, perm):float()

model = torch.load('logs/kmeans/model.net0.735')
last_layer = model:size()
model:remove(last_layer)

model:evaluate()

local bs = 25
error_count = 0
linear_output = torch.Tensor(1000, 1024)
for i=1,X:size(1),bs do
  local outputs = model:forward(X:narrow(1,i,bs):cuda())
  for j=1,outputs:size(1) do
    cIndex = (i-1) + j
    linear_output[cIndex] = outputs[j]:float()
  end
end

print("starting tsne")
opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(X, opts)
print("Generating the image")

im_size = 4096
map_im = m.draw_image_map(mapped_x1, X, im_size, 0, true)

print("Saving image")
image.save('tsne.png', map_im)
