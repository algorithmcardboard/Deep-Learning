require 'torch';
stringx = require('pl.stringx')
require 'io'
ptb = require('data')

torch.setdefaulttensortype('torch.FloatTensor')

local ptb_path = "./data/"

local trainfn = ptb_path .. "ptb.train.txt"
local testfn  = ptb_path .. "ptb.test.txt"
local validfn = ptb_path .. "ptb.valid.txt"

require 'base';
require 'nn';
require 'nngraph';

params = {}
params.rnn_size = 200
params.layers = 2
params.gru = false
params.batch_size = 20
model = {}


model = torch.load('/scratch/ajr619/lstm/model.net120.10116158295')
-- model.core_network = torch.load('core_network.net')  -- the core RNN

vocab_map = {}
inverse_map = {}

function initialize_states()
    model.s = {}
    for j = 0, 1 do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
          model.s[j][d] = torch.zeros(1, params.rnn_size)
        end
    end

    ptb.traindataset(params.batch_size)
    ptb.traindataset(params.batch_size)
    ptb.testdataset(params.batch_size)
    vocab_map = ptb.vocab_map
    inverse_map = ptb.inverse_map
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if vocab_map[line[i]] == nil then line[i] = '<unk>' end
  end
  return line
end

print ("Building vocab and initialize states")

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
  for i = 2, #line-2 do
    local x = torch.Tensor(1):fill(vocab_map[line[i]])
    local y = torch.Tensor(1):fill(vocab_map[line[i + 1]])
    
    local nll = nil
    perp_temp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    g_replace_table(model.s[0], model.s[1])
  end
  -- make model generate more
  local x = torch.Tensor(1):fill(vocab_map[line[#line]])
  local y = torch.Tensor(1):fill(vocab_map[line[#line]])
  local next_words = {}
  for i = 1, n_words  do 
    -- doesn't matter what y is since we're just generating predictions.
    -- we just need the x's and s's
    local s = model.s[0]
    local nll = nil
    nll, model.s[1] = unpack(model.core_network:forward({x, y, s}))
    
    -- this is the log-softmax layer
    -- model.core_network:get(44) is the log softmax layer
    local logsoftmax = nil
    local word_ind = nil
    logsoftmax, word_ind = model.core_network:get(44).output:max(2)
    local next_word = inverse_map[word_ind[1][1]]
    next_words[i] = next_word
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

g_disable_dropout(model.rnns)

-- set up stuff.
initialize_states()

query_sentences()

