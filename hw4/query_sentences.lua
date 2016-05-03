require 'base';
require 'nn';
require 'nngraph';
require 'torch';
stringx = require('pl.stringx')
require 'io'
ptb = require('data')

params = {}
params.rnn_size = 200
params.layers = 2
params.gru = false
params.seq_length = 20

model = {}
model.s = {}
model.s[0] = {}
for d = 1, 2 * params.layers do
  model.s[0][d] = torch.zeros(1, params.rnn_size)
end

--print(model.s)

model_file = "/scratch/ajr619/lstm/model.net120.10116158295"
print ("Loading model "..model_file)
m = torch.load(model_file)
model.core_network = m.core_network

print ("Building vocab and initialize states")
local trainfn = "data/ptb.train.txt"
local testfn  = "data/ptb.test.txt"
local validfn = "data/ptb.valid.txt"

ptb.load_data(trainfn)
ptb.load_data(validfn)
ptb.load_data(testfn)

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if ptb.vocab_map[line[i]] == nil then line[i] = '<unk>' end
  end
  return line
end


function getline()
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
        return line.code
    elseif line.code == "vocab" then
      print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    return line
  end
end

function get_next_words(line)
  local n_words = tonumber(line[1])
  -- Fillup hidden states with known words
  local x
  local y
  x = torch.Tensor(1):fill(ptb.vocab_map[line[#line - 1]])
  y = torch.Tensor(1):fill(ptb.vocab_map[line[#line]])
  for i = 2, #line-3 do
    x = torch.Tensor(1):fill(ptb.vocab_map[line[i]])
    y = torch.Tensor(1):fill(ptb.vocab_map[line[i + 1]])

    -- print ('x is ', x)
    -- print ('y is ', y)
    
    output = model.core_network:forward({x, y, model.s[0]})
    -- print (output)
    perp_temp, model.s[1] = unpack(output)
    g_replace_table(model.s[0], model.s[1])
  end
  -- make model generate more
  -- local x = torch.Tensor(1):fill(ptb.vocab_map[line[#line]])
  -- local y = torch.Tensor(1):fill(ptb.vocab_map[line[#line]])
  local next_words = {}
  -- print(x)
  -- print(y)
  for i = 1, n_words  do 
    -- doesn't matter what y is since we're just generating predictions.
    -- we just need the x's and s's
    perp_temp, model.s[1] = unpack(model.core_network:forward({x, y, model.s[0]}))
    
    local logsoftmax = nil
    local word_ind = nil
    --logsoftmax, word_ind = model.core_network.modules[44].output:max(2)
    word_ind = torch.multinomial(torch.exp(model.core_network.modules[44].output), 1)
    local next_word = ptb.inverse_map[word_ind[1][1]]
    -- print('word index is ', word_ind, ' word is ', next_word)
    next_words[i] = next_word
    x = y
    y = torch.Tensor(1):fill(ptb.vocab_map[next_word])
    g_replace_table(model.s[0], model.s[1])
    
  end
  return next_words
end


function query_sentences()
  while true do
    print("Query: len word1 word2 etc")
    local line = getline()
    if line == "EOF" then
      break
    else
      next_words = get_next_words(line)
      
      line = stringx.join(" ", line)
      next_words = stringx.join(" ", next_words)
      print(line .." ".. next_words)
    end
  end
end

function disable_dropout(node)
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end

g_disable_dropout(model.core_network)

query_sentences()
