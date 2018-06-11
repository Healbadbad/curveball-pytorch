import torch
from torch.autograd import Variable
from torch.autograd import grad
from curveball import Curveball

def rosenbrock(u, v, noise_range=1):
    noise = torch.rand(1) * noise_range + 1
    return (1-u) ** 2 + 100 * noise * (v - u ** 2)**2

# SGD
print("--------- SGD  ---------")

uv = torch.tensor([-0.25, -0.25], requires_grad = True)

optimizer = torch.optim.SGD([uv], lr = 0.001)
i = 0
loss = 99999
while loss > 0.005:
    i+=1
    optimizer.zero_grad()
    loss = rosenbrock(uv[0], uv[1])
    #loss_sum.backward(torch.ones_like(w))
    loss.backward()
    optimizer.step()
print("SGD iterations:", i)
print("uv=", uv)
sgduv = uv
#################################################################
# Curveball
print("--------- Curveball  ---------")
uv = torch.tensor([-0.25, -0.25], requires_grad = True)
optimizer = Curveball([uv], lr = 1, momentum=0.9) # 1, 0.9

j = 0
loss = 99999
while loss > 0.005:
    j+=1
    optimizer.zero_grad()
    loss = rosenbrock(uv[0], uv[1])

    print("LOSS: ------------------", loss)
    loss.backward(retain_graph=True)
    optimizer.step(uv, lambda: loss)
    print("new u, v: ", uv.data)

print("SGD iterations:", i)
print("sgduv=", sgduv)
print("Curvball iterations:", j)
print("uv=", uv)
