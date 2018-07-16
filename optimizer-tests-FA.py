import torch
from torch.autograd import Variable
from torch.autograd import grad
from curveball import Curveball

def rosenbrock(u, v, noise_range=0):
    noise = torch.rand(1) * noise_range + 1
    return (1-u) ** 2 + 100 * noise * (v - u ** 2)**2

# SGD
print("--------- SGD  ---------")

#uv = torch.tensor([-0.25, -0.25], requires_grad = True)

u = torch.tensor([-0.25], requires_grad = True)
v = torch.tensor([-0.25], requires_grad = True)
optimizer = torch.optim.SGD([u], lr = 0.001)
i = 0
loss = 99999
#while loss > 0.005:
#    i+=1
for i in range(3000):
    optimizer.zero_grad()
    loss = rosenbrock(u, v)
    #loss_sum.backward(torch.ones_like(w))
    loss.backward()
    optimizer.step()
print("SGD iterations:", i)
#print("uv=", uv)
print("u, v: ", u, v)
sgduv = [u, v]
#################################################################
# Curveball
print("--------- Curveball  ---------")
u = torch.tensor([[0.25]], requires_grad = True)
v = torch.tensor([[0.25]], requires_grad = True)
uv = torch.tensor([[-0.25], [-0.25]], requires_grad = True)
optimizer = Curveball([u, v], lr = 1, momentum=0.9) # 1, 0.9



grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

j = 0
loss = 99999
#while loss > 0.005:
#    j+=1

for j in range(50):
    optimizer.zero_grad()
    #intermediate = torch.tensor(out**2, requires_grad = True)
    intermediate = torch.cat([u, v]) #, requires_grad=True)
    print(intermediate)
    #loss = rosenbrock(intermediate[0], intermediate[1]).view(1,-1)
    loss = rosenbrock(intermediate[0], intermediate[1]).view(1,-1)

    print("LOSS: ------------------", loss)
    print("intermediate: ", intermediate)
    print("loss grad:", loss.grad)
    #loss.backward(retain_graph=True)
    print("loss grad:", loss.grad)

    #y_onehot = torch.FloatTensor( out.size(1))
    #y_onehot.zero_()

    #jacobian_model = torch.zeros(intermediate.size(1), uv.size(0))
    print("intermediate grad:", intermediate.grad)
    intermediate.register_hook(save_grad("intermediate"))
    #for k in range(intermediate.size(1)):
    #    intermediate.backward(y_onehot.scatter(0,torch.tensor(k),1), retain_graph=True) 
    #    jacobian_model[k] = uv.grad
    #    print("int grad, k=",k, grads["intermediate"])
    #    uv.grad.data.zero_()
    #    optimizer.zero_grad()
    #grad_loss = grad(loss, intermediate, create_graph=True, retain_graph=True)[0]
    #grad_model = grad(intermediate, uv, grad_loss, create_graph=True, retain_graph=True)[0]

    #print("grad_loss top: ", grad_loss)
    #print("grad_model top: ", grad_model)
    #print("jacobian_model top: ", jacobian_model)
    #print("uv grad top: ", uv.grad)
    #hessian_maybe = torch.stack([grad(grad_loss[:, i], intermediate, create_graph=True, retain_graph=True)[0] for i in range(out.size(1))], dim=-1)[0]
    #
    #print("hessian maybe :", hessian_maybe)
    #loss.register_hook(save_grad("loss"))
    print(loss.backward(retain_graph=True, create_graph=True))
    #print("---- grads ----")
    print("loss grad:", loss.grad)
    #print("uv grad:", uv.grad)
    #print("intermediate grad:", intermediate.grad)
    #print("intermediate grad:", grads["intermediate"])
    #print("loss grad:", grads["loss"])
    print("u grad: ", u.grad)
    print("v grad: ", v.grad)
    print("intermediate grad:", intermediate.grad)
    print("intermediate grad:", grads["intermediate"])

    optimizer.step(intermediate, grads["intermediate"], lambda: loss)
    #print("loss grad:", grads["loss"])
    print("*********** new u, v: ", u, v)
    print("*********** new u, v: ", uv)

print("SGD iterations:", i)
print("sgduv=", sgduv)
print("Curvball iterations:", j)
print("*********** final u, v: ", u, v)
#print("uv=", uv)
