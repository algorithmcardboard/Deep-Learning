require 'torch'
require 'nn'
require 'nngraph'

--equations
x = nn.Linear(4, 2)()
y = nn.Linear(5, 2)()
z = nn.Identity()()

h1 = nn.Square()(nn.Tanh()(x))
h2 = nn.Square()(nn.Sigmoid()(y))
h3 = nn.CMulTable()({h1,h2})

h4 = nn.CAddTable()({h3,z})

--build gModule
m = nn.gModule({x,y,z}, {h4})
-- graph.dot(m.fg, 'MLP','gModule')

--inputs
in1 = torch.Tensor({1, 2, 3, 4})
in2 = torch.Tensor({1, 2, 3, 4, 5})
in3 = torch.Tensor({1, 2})

print("Outputs are")
print(m:forward({in1, in2, in3}))

gradient = m:backward({in1, in2, in3}, torch.ones(2))

print("Gradients are ")
for i = 1, 3 do
  print(gradient[i])
end
