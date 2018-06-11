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
        self.B = 0.1
        self.momentum = momentum
        super(Curveball, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Curveball, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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

    def delta_z(self, output, loss, model_params, z, param_lambda=1):
        ''' Hessian Vector Product:
            Effectively d_z = hvp(buf) + Jacobian '''

        grads_model_parts = [grad(out, model_params, create_graph=True)[0] for out in output]
        grads_model = torch.stack(grads_model_parts) 
        grad_loss = grad(loss, output, create_graph=True, retain_graph=True)[0]
        #hessian_parts = [grad(grad_loss_component, model_params, retain_graph=True)[0] for grad_loss_component in grad_loss]
        hessian_parts = [grad(grad_loss_component, output, retain_graph=True)[0] for grad_loss_component in grad_loss]
        hessian_loss = torch.stack(hessian_parts) 
        print("grads_model: ", grads_model)
        print("grads_loss: ", grad_loss)
        print("hessian_loss: ", hessian_loss)
        print("grads_model.t: ", grads_model.t())
        print("z: ", z)
        z_var = torch.autograd.Variable(z).view(-1,1)
        d_z = hessian_loss.mm(grads_model.t().mm(z_var))
        print("d_z: ", d_z)
        d_z = grads_model.mm(d_z + grad_loss.view(-1,1))
        print("d_z #2 :", d_z)
        d_z = d_z + param_lambda * z_var
        print("d_z #3 :", d_z)
        try:
            self.B, self.momentum = self.autotune(z_var, d_z, hessian_loss, grads_model, grad_loss)
        except Exception as ex:
            print(ex)
            pass
        print("B,momentum are: ", self.B, self.momentum)
        return d_z.data

    def autotune(self, z, dz, Hl, Jmodel, Jloss):

        print("---- Autotuning ----")
        print("Jmodel: ", Jmodel)
        print("Jloss: ", Jloss.view(-1,1))
        print("z: ", z)
        J = Jmodel.mm(Jloss.view(-1,1))
        Hhat = Jmodel.mm(Hl.mm(Jmodel.t()))
        print("0,0 :", dz.t().mm(Hhat.mm(dz)))
        print("0,1 :", z.t().mm(Hhat.mm(dz)))
        print("1,0 :", z.t().mm(Hhat.mm(dz)))
        print("1,1 :", z.t().mm(Hhat.mm(z)))
        part1 = torch.Tensor([[-dz.t().mm(Hhat.mm(dz)), -z.t().mm(Hhat.mm(dz))],
                               [-z.t().mm(Hhat.mm(dz)), -z.t().mm(Hhat.mm(z))]] )
        invp1 = part1.inverse()
        part2 = torch.tensor([[J.t().mm(dz)],[J.t().mm(z)]])
        answers = invp1.mm(part2)
        print("answers: ", answers)
        print("answers[0]: ", answers[0])

        return -answers[0], answers[1]



    def step(self, output, closure=None):
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
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                #if momentum != 0:
                param_state = self.state[p]
                momentum = self.momentum

                split = True
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    #print("here")
                    
                    if split:
                        d_buf = self.delta_z(output, loss, p, buf)
                        print("buf:", buf)
                        print("d_buf:", d_buf)
                        print("-B * d_buf:", -self.B * d_buf)
                        buf.mul_(self.momentum)
                        buf.add_(-self.B * d_buf.view(-1))
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
                        d_buf = self.delta_z(output, loss, p, buf)
                        print("buf: ", buf)
                        print("-B * d_buf:", -self.B * d_buf)
                        print("-B * d_buf:", -self.B * d_buf.view(-1))
                        print("newbuf: ", buf.add(-self.B * d_buf.view(-1)))
                        buf.mul_(self.momentum)
                        buf.add_(1 - dampening, -self.B * d_buf.view(-1))
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
