require 'nngraph'
require('base')
ptb = require('data')

MODEL_FILE = '/scratch/ajr619/lstm/model.net120.10116158295'

print('Loading model file '.. MODEL_FILE)
model = torch.load(MODEL_FILE)

params = { 
  layers = 2,
  batch_size = 20
}

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2*params.layers do
            model.start_s[d]:zero()
        end
    end
end

function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        if i % 1000  == 0 then
          print("Iterating "..i .. " " .. len)
        end
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

ptb.traindataset(params.batch_size);
ptb.validdataset(params.batch_size);

state_test =  {data=ptb.testdataset(params.batch_size)}

print("Running test set")
run_test()
