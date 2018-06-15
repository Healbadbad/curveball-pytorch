import torch
from torch.autograd import Variable
from torch.autograd import grad
from curveball import Curveball

def rosenbrock(u, v, noise_range=0):
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


grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

j = 0
loss = 99999
while loss > 0.005:
    j+=1

#for j in range(50):
    optimizer.zero_grad()
    out = uv.view(1,-1)
    #intermediate = torch.tensor(out**2, requires_grad = True)
    intermediate = out
    loss = rosenbrock(intermediate[0, 0], intermediate[0, 1]).view(1,-1)

    print("LOSS: ------------------", loss)
    print("intermediate: ", intermediate)
    print("loss grad:", loss.grad)
    print("uv grad:", uv.grad)
    #loss.backward(retain_graph=True)
    print("loss grad:", loss.grad)
    print("uv grad:", uv.grad)

    #out = torch.tensor([[uv[0], uv[1]]])
    print("out size:", out.size())
    print("out size:", out.size(1))
    #grad_loss = grad(loss, uv,  create_graph=True, retain_graph=True)[0]
    
    #grad_model = grad(intermediate, uv, create_graph=True, retain_graph=True)[0]
    #jacobian_model = torch.stack([grad(out[:, i], uv, create_graph=True, retain_graph=True)[0] for i in range(out.size(1))], dim=-1)

    #for k in range(out.size(1)):
    #jacobian_model = torch.stack([grad(out[:, i], uv, create_graph=True, retain_graph=True)[0] for i in range(out.size(1))], dim=-1)
    y_onehot = torch.FloatTensor( out.size(1))
    y_onehot.zero_()

    jacobian_model = torch.zeros(intermediate.size(1), uv.size(0))
    intermediate.register_hook(save_grad("intermediate"))
    for k in range(intermediate.size(1)):
        intermediate.backward(y_onehot.scatter(0,torch.tensor(k),1), retain_graph=True) 
        jacobian_model[k] = uv.grad
        print("int grad, k=",k, grads["intermediate"])
        uv.grad.data.zero_()
        optimizer.zero_grad()
    grad_loss = grad(loss, intermediate, create_graph=True, retain_graph=True)[0]
    grad_model = grad(intermediate, uv, grad_loss, create_graph=True, retain_graph=True)[0]

    print("grad_loss top: ", grad_loss)
    print("grad_model top: ", grad_model)
    print("jacobian_model top: ", jacobian_model)
    print("uv grad top: ", uv.grad)
    #hessian_maybe = grad(grad_loss, out, retain_graph=True)
    hessian_maybe = torch.stack([grad(grad_loss[:, i], intermediate, create_graph=True, retain_graph=True)[0] for i in range(out.size(1))], dim=-1)[0]
    #uv.grad.backward(torch.ones_like(uv))
    #real_hessian = grad(grads["intermediate"], intermediate, create_graph=True, retain_graph=True)
    
    print("hessian maybe :", hessian_maybe)
    #print("real hessian :", real_hessian)
    #optimizer.zero_grad()
    loss.register_hook(save_grad("loss"))
    loss.backward(retain_graph=True, create_graph=True)
    print("---- grads ----")
    print("loss grad:", loss.grad)
    print("uv grad:", uv.grad)
    print("intermediate grad:", intermediate.grad)
    print("intermediate grad:", grads["intermediate"])
    print("loss grad:", grads["loss"])
    optimizer.step(intermediate, grads["intermediate"], lambda: loss)
    print("loss grad:", grads["loss"])
    print("new u, v: ", uv.data)

print("SGD iterations:", i)
print("sgduv=", sgduv)
print("Curvball iterations:", j)
print("uv=", uv)
