require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'optim'   -- an optimization package, for online and batch methods

model_path = "results/model.net.39"

model = torch.load(model_path)

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
data_path = 'mnist.t7'
train_file = paths.concat(data_path, 'train_32x32.t7')
test_file = paths.concat(data_path, 'test_32x32.t7')

if not paths.filep(train_file) or not paths.filep(test_file) then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

trsize = 60000
tesize = 10000

trsize = 6000
tesize = 1000

loaded = torch.load(train_file, 'ascii')
trainData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return trsize end
}

loaded = torch.load(test_file, 'ascii')
testData = {
   data = loaded.data,
   labels = loaded.labels,
   size = function() return tesize end
}

trainData.data = trainData.data:float()
testData.data = testData.data:float()

print '==> preprocessing data: normalize globally'
mean = trainData.data[{ {},1,{},{} }]:mean()
std = trainData.data[{ {},1,{},{} }]:std()

-- trainData.data[{ {},1,{},{} }]:add(-mean)
-- trainData.data[{ {},1,{},{} }]:div(std)

-- Normalize test data, using the training means/stds
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)

trainMean = trainData.data[{ {},1 }]:mean()
trainStd = trainData.data[{ {},1 }]:std()

testMean = testData.data[{ {},1 }]:mean()
testStd = testData.data[{ {},1 }]:std()

cmd = torch.CmdLine()
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:text()

opt = cmd:parse(arg or {})

classes = {'1','2','3','4','5','6','7','8','9','0'}
confusion = optim.ConfusionMatrix(classes)

dofile '5_test.lua'

_prediction = torch.FloatTensor()
_max = torch.FloatTensor()
_pred_idx = torch.LongTensor()
_targ_idx = torch.LongTensor()

_prediction:resize(pred:size()):copy(pred)

_max:max(self._pred_idx, self._prediction, 1)

observed_class = _pred_idx[1]

test()
