require 'nn'

I = torch.Tensor({4, 5, 2, 2, 1, 3, 3, 2, 2, 4, 4, 3, 4, 1, 1, 5, 1, 4, 1, 2, 5, 1, 3, 1, 4}):reshape(1, 5,5)

W = torch.Tensor({4, 3, 3, 5, 5, 5, 2, 4, 3}):reshape(3,3)

s = nn.SpatialConvolution(1, 1, 3, 3)

s.weight = W

output = s:forward(I)

gradients = s:backward(I, torch.ones(1, 3,3))
print(gradients)
