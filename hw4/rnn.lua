local sentence, ws = {"the", "ball", "is", "red"}, {} 
for i,v in ipairs (sentence) do ws [ i ] = torch.LongTensor {w2i[v]} end 

-- Parameters of the RNN (cf. slides) 
local R = nn.Linear ( nhid , nhid) 
local A = nn.LookupTable (#i2w, nhid) 
local B = nn.Linear ( nhid , #i2w) 

-- one column [h(t-1) ,w(t) ] -> [h(t), y(t)] 
local m       = nn.Sequential () 
local encoder = nn.ParallelTable ():add(R):add(A) 
local decoder = nn.ConcatTable ():add( nn.Identity ()):add(B) m:add (encoder):add( nn.CAddTable ()):add( nn.Sigmoid ()):add(decoder) 

-- create the RNN by cloning gradInput and output 
local rnn = {} 
for t = 1, # ws - 1 do 
  rnn [t] = m:clone {'weight',' gradWeight ','bias',' gradBias '} 
end 

-- Forward: 
local hs , ys , out = {[0] = torch.Tensor ( nhid):zero()}, {} 
for t = 1, # ws - 1 do 
  out = rnn [t]:forward{ hiddens [t-1], ws [t]} 
  table.insert ( hs , out[1]) 
  table.insert ( ys , out[2]) 
end 

-- Backward: 
local gradh , grad, err = torch.Tensor ( nhid):zero() 
for t = # ws - 1, 1, -1 do 
criterion:forward ( ys [t], ws [t+1]) 
  err   = criterion:backward ( ys [t], ws [t+1]) 
  grad  = rnn [t]:backward({ hs [t-1], ws [t]}, { gradh , err}) 
  gradh = grad[1] 
  gradh:div ( math.max (10, gradh:norm (2))) -- clipping 
end 
