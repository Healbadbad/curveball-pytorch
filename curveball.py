import torch
#from .optimizer import Optimizer, required
from torch.optim.optimizer import Optimizer, required
from torch.autograd import grad


class Curveball(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.Curveball(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        #TODO: Change to use their interface
        self.B = 0.001
        self.momentum = momentum
        self.grads = {}
        super(Curveball, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Curveball, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def hvp(self, y, x, v):
        v.requires_grad = False
        grad_result = grad(y, x, create_graph=True)[0]
        elemwise_prods = grad_result * v
        elemwise_prods.backward(torch.ones_like(x), retain_graph=True) # by evaluating this at 1, get x grad
        return x.grad

    def delta_z_no_split(self, loss, model_params, z, param_lambda=0):
        ''' Hessian Vector Product:
            Effectively d_z = hvp(buf) + Jacobian '''
        print("---- d_z_no_split ----")
        print("loss:", loss)
        print("model_params:", model_params)
        print("model_params.grad:", model_params.grad)
        print("z: ", z)
        grad_loss = grad(loss, model_params, create_graph=True, retain_graph=True)[0]
        print("model_params.grad:", model_params.grad)
        print("grad_loss: ", grad_loss)
        d_z = self.hvp(loss, model_params, z)
        print("dz from hvp: ", d_z)
        d_z = d_z + grad_loss

        return d_z.data

    def Lop(self, ys, xs, ws):
        return torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)


    def Rop(self, ys, xs, vs):
        # Computes the Jacobian-Vector Product
        # By doing two backwards passes

        if isinstance(ys, tuple):
            ws = [torch.tensor(torch.zeros(y.size()), requires_grad=True) for y in ys]
        else:
            ws = torch.tensor(torch.zeros(ys.size()), requires_grad=True)

        gs = torch.autograd.grad(ys, xs, grad_outputs=ws, create_graph=True, retain_graph=True, allow_unused=True)
        print("gs: ", gs)
        print("ws: ", ws)
        print("vs: ", vs)
        
        re = torch.autograd.grad(gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=True)
        print("re: ", re)
        return re


    def delta_z(self, output, output_grad, loss, model_params, z, param_lambda=0):
        ''' Hessian Vector Product:
            Effectively d_z = hvp(buf) + Jacobian '''


        # 2 passes are needed to compute Hz
        # 1 forward to compute Jmz
        #   Implemented as Two backwards passes 
        #   https://j-towns.github.io/2017/06/12/A-new-trick.html
        # Manually compute the Hessian Hl
        # manually compute v' = HlJmz
        # 1 backwards pass to compute Jm.t()(v')
        #   As output.backward(v')
        # Note: Computing HlJmz can also be done with a backwards pass 
        #   On a small part of the graph - loss to output of model

        # TODO: Refactor computing Hz

        # Hd_z

        #grad_model = grad(output, model_params, create_graph=True, retain_graph=True)[0]
        #grad_loss = grad(loss, output, create_graph=True, retain_graph=True)[0]
        print("---- Delta Z ----:")
        print("model params: ", model_params)
        print("model params grad: ", model_params.grad)
        output_var = torch.tensor(output, requires_grad=True)
        grad_loss = torch.tensor(output_grad.view(-1,1), requires_grad=True)
        grad_model = model_params.grad.data.view(-1,1)
        z_var = torch.autograd.Variable(z).view(-1,1)
        print("z: ", z)
        print("z_var: ", z_var)
        Jmz = self.Rop(output, model_params, z)[0].view(-1,1)
        print("Jmz from Rop: ", Jmz)

        #hessian_loss = torch.stack(hessian_parts) 

        print("output: ", output)
        print("output size: ", output.size())
        print("grad_loss: ", grad_loss)
        print("grad_loss size: ", grad_loss.size())
        print("grad_model: ", grad_model)
        print("grad_model.t: ", grad_model.t())
        print(grad_loss.t()[:, 0])

        # May be able to compute Hl more efficiently?

        hessian_loss = torch.stack([grad(grad_loss.t()[:, i], output, create_graph=True, retain_graph=True)[0].view(-1) for i in range(output.size(0))], dim=-1)
        print("hessian_loss: ", hessian_loss)
        print("Hessian shae: ", hessian_loss.size())
        print("Jmz shape: ", Jmz.size())
        #HlJmz = torch.tensor(hessian_loss.mm(Jmz) + grad_loss, requires_grad=True)
        HlJmz = hessian_loss.mm(Jmz)
        HlJmzJl = HlJmz + grad_loss
        print("HlJmz: ", HlJmz)
        print("HlJmzJl: ", HlJmzJl)



        #Todo: back pass to compute Jm'v'
        print("Model_params.grad before backpass", model_params.grad)
        #output.backward(HlJmz)
        print("HlJmz size: ", HlJmz.size())

        #gs = grad(output_var, model_params.view(-1,1), grad_outputs=HlJmz, create_graph=True, retain_graph=True)   
        print("model_params new view ", model_params.view(1,-1))
        #gs = self.Lop(output, model_params.view(1,-1), HlJmz.view(1,-1))
        #Hhatz = self.Lop(output_var, model_params, HlJmz.view(-1))[0].view(-1,1) + param_lambda * z_var
        Hhatz = self.Lop(output_var, model_params, HlJmz.view(-1))[0].view(-1,1) + param_lambda * z_var
        gs = self.Lop(output_var, model_params, HlJmzJl.view(-1))[0].view(-1,1)
        print("gs: ", gs)
        #output.backward(HlJmz, create_graph=True, retain_graph=True)


        print("Model_params.grad after backpass", model_params.grad)

        d_z = gs + param_lambda * z_var
        print("d_z: ", d_z)

        Jmdz = self.Rop(output, model_params, d_z)[0].view(-1,1)
        print("Jmdz: ", Jmdz)
        HlJmdz = hessian_loss.mm(Jmdz)
        Hhatdz = self.Lop(output_var, model_params, HlJmdz.view(-1))[0].view(-1,1) + param_lambda * d_z
        
        
        print("Hhatz: ", Hhatz)
        print("Hhatdz: ", Hhatdz)
        print("J: ", model_params.grad)

        #d_z = output.grad
        #d_z = grads["output"]
        #d_z = hessian_loss.mm(grad_model.t().mm(z_var))
        #d_z = grad_loss.grad
        #print("d_z: ", d_z)

        #d_z = grad_model.mm(d_z + grad_loss)
        #print("d_z #2 :", d_z)
        #d_z = d_z + param_lambda * z_var
        #print("d_z #3 :", d_z)
        #Jz = self.Lop(loss, model_params, z)[0].view(-1,1) 
        Jtz = model_params.grad.t() @ z
        #Jdz = self.Lop(loss, model_params, d_z)[0].view(-1,1) 
        Jtdz = model_params.grad.t() @ d_z
        # Autotune needs Hz, Hdz
        try:
            #self.B, self.momentum = self.autotune(z_var, d_z, Hhatz, Hhatdz, model_params.grad.view(1,-1))
            self.B, self.momentum = self.autotune(z_var, d_z, Hhatz, Hhatdz, Jtz, Jtdz)
        except Exception as ex:
            print(ex)
            pass
        print("B,momentum are: ", self.B, self.momentum)
        return d_z.data

    #def autotune(self, z, dz, Hhatz, Hhatdz, Jt):
    def autotune(self, z, dz, Hhatz, Hhatdz, Jz, Jdz):

        print("---- Autotuning ----")
        print("z: ", z)
        #J = Jmodel.mm(Jloss.view(-1,1))
        # J'z = model_params.grad' @ z 
        # if loss is a scalar model_params.grad == J
        print("dz: ", dz)
        print("dz.t: ", dz.t())
        #print("0,0 :", dz.t().mm(dz))
        print("0,0 :", dz.t().mm(Hhatdz))
        print("0,1 :", z.t().mm(Hhatdz))
        print("1,0 :", z.t().mm(Hhatdz))
        print("1,1 :", z.t().mm(Hhatz))
        part1 = torch.Tensor([[-dz.t().mm(Hhatdz), -z.t().mm(Hhatdz)],
                               [-z.t().mm(Hhatdz), -z.t().mm(Hhatz)]] )
        invp1 = part1.inverse()
        print("here?")
        #part2 = torch.tensor([[Jt.mm(dz)],[Jt.mm(z)]])
        part2 = torch.tensor([[Jdz],[Jz]])
        print("part1: ", part1)
        print("invp1: ", invp1)
        print("part2: ", part2)
        answers = invp1.mm(part2)
        print("answers: ", answers)
        print("answers[0]: ", answers[0])

        return -answers[0], answers[1]



    def step(self, output, output_grad, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        B = self.B 
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                # Not Necessary?
                #if p.grad is None:
                #    continue
                #d_p = p.grad.data
                #if weight_decay != 0:
                #    d_p.add_(weight_decay, p.data)
                #if momentum != 0:
                param_state = self.state[p]
                momentum = self.momentum

                split = True
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    #print("here")
                    
                    if split:
                        d_buf = self.delta_z(output, output_grad, loss, p, buf)
                        print("buf:", buf)
                        print("d_buf:", d_buf)
                        print("-B * d_buf:", -self.B * d_buf)
                        buf.mul_(self.momentum)
                        buf.add_(-self.B * d_buf)
                    else:
                        d_buf = self.delta_z_no_split(loss, p, buf)
                        print("buf:", buf)
                        print("d_buf:", d_buf)
                        buf.mul_(self.momentum)
                        buf.add_(-self.B * d_buf)
                    #buf.add_(-B * d_buf)
                    #buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state['momentum_buffer']
                    if split:
                        d_buf = self.delta_z(output, output_grad, loss, p, buf)
                        print("buf: ", buf)
                        print("-B * d_buf:", -self.B * d_buf)
                        print("newbuf: ", buf.add(-self.B * d_buf))
                        buf.mul_(self.momentum)
                        buf.add_(1 - dampening, -self.B * d_buf)
                    else:
                        buf.mul_(self.momentum)
                        d_buf = self.delta_z_no_split(loss, p, buf)
                        buf.add_(1 - dampening, -self.B * d_buf)
                    #buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf


                p.data.add_(group['lr'], d_p)

        return loss
