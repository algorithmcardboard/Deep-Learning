require 'base';
require 'nn';
require 'nngraph';
require 'torch';
require 'io'
stringx = require('pl.stringx')
ptb = require('data')


function readline()
  local line = io.read("*line")

  if line == nil or #line == 0 then error({code="EOF"}) end
  line = stringx.split(line)
  if line == nil then error({code="EOF"}) end
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
        print ('code is EOF')
        return line.code
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


function get_model()
  model = {
    s = {
      [0] = {}
    }
  }
  for d = 1, 2 * params.layers do
    model.s[0][d] = torch.zeros(1, params.rnn_size)
  end


  model.core_network = torch.load(MODEL_FILE).core_network
  g_disable_dropout(model.core_network)

  return model
end


function get_vocab_inverse_map()
  local trainfn = "data/ptb.train.txt"
  local testfn  = "data/ptb.test.txt"
  local validfn = "data/ptb.valid.txt"
  ptb.load_data(trainfn)
  ptb.load_data(validfn)
  ptb.load_data(testfn)

  return ptb.vocab_map, ptb.inverse_map
end


function get_next_words(line)
  local n_words = tonumber(line[1])
  -- initialize x and y to the penultimate and last words
  local x = torch.Tensor(1):fill(ptb.vocab_map[line[#line - 1]])
  local y = torch.Tensor(1):fill(ptb.vocab_map[line[#line]])

  for i = 2, #line-3 do
    x = torch.Tensor(1):fill(ptb.vocab_map[line[i]])
    y = torch.Tensor(1):fill(ptb.vocab_map[line[i + 1]])

    perp_temp, model.s[1] = unpack(model.core_network:forward({x, y, model.s[0]}))
    g_replace_table(model.s[0], model.s[1])
  end

  local next_words = {}
  for i = 1, n_words  do 
    perp_temp, model.s[1] = unpack(model.core_network:forward({x, y, model.s[0]}))
    
    local word_ind = torch.multinomial(torch.exp(model.core_network.modules[44].output), 1)
    local word = ptb.inverse_map[word_ind[1][1]]
    next_words[i] = word

    -- swap
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
      
      table.remove(line, 1)
      line = stringx.join(" ", line)
      next_words = stringx.join(" ", next_words)
      print(line .." ".. next_words)
    end
  end
end

params = {
  rnn_size = 200,
  layers = 2,
  gru = false,
  seq_length = 20
}

print ("Loading model "..MODEL_FILE)

MODEL_FILE = "/scratch/ajr619/lstm/model.net120.10116158295"
model = get_model()

print("Building vocab and initialize states")
vocab_map, inverse_map = get_vocab_inverse_map()

-- Querying sentences
query_sentences()
