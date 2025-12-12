import torch
from torch.optim import Optimizer

# Momentum SGD that uses Strong Wolfe line search to determine step length
class MomentumSGD_Strong_Wolfe(Optimizer):
    def __init__(self, params, alpha1=0.01, momentum=0.9, c1=1e-4, c2=0.9, 
                 alpha_max=0.1, max_ls_iter=100):
        defaults = dict(alpha1=alpha1, momentum=momentum, c1=c1, c2=c2,
                        alpha_max=alpha_max, max_ls_iter=max_ls_iter)
        super().__init__(params, defaults)
    
    # obtains a flattened view of all parameters
    def _gather_flat_params(self):
        return torch.cat([p.data.view(-1) for group in self.param_groups for p in group['params']]) 
    
    # obtains a flattened view of all gradients
    def _gather_flat_grad(self):
        return torch.cat([p.grad.view(-1) for group in self.param_groups for p in group['params']])
    
    # sets the parameters of the model from a flattened view
    def _set_params_from_flat(self, flat):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(flat[idx:idx+numel].view_as(p))
                idx += numel

    # f(alpha) = loss at x0 + alpha * d
    def _phi(self, closure, x0, d, alpha):
        self._set_params_from_flat(x0 + alpha * d)
        loss = closure()
        g = self._gather_flat_grad()
        return loss.item(), g
    
    # derivative of phi at alpha
    def _phi_derivative(self, grad, d):
        return torch.dot(grad, d).item()

    # zoom phase of Strong Wolfe line search
    def _zoom(self, closure, x0, d, alo, ahi, phi0, der0, c1, c2, max_ls_iter):
        for _ in range(max_ls_iter):
            aj = 0.5 * (alo + ahi)
            phi_j, g_j = self._phi(closure, x0, d, aj)
            if phi_j > phi0 + c1 * aj * der0:
                ahi = aj
            else:
                der_j = self._phi_derivative(g_j, d)
                if abs(der_j) <= -c2 * der0:
                    return aj
                if der_j * (ahi - alo) >= 0:
                    ahi = alo
                alo = aj
        return aj

    # Strong Wolfe line search
    def _strong_wolfe(self, closure, x0, d, phi0, der0, 
                      c1, c2, alpha1, alpha_max, max_ls_iter):
        alpha_prev = 0.0
        phi_prev = phi0

        alpha = alpha1
        for _ in range(max_ls_iter):
            phi_a, g_a = self._phi(closure, x0, d, alpha)

            if (phi_a > phi0 + c1 * alpha * der0) or \
               (phi_a >= phi_prev and _ > 0):
                return self._zoom(closure, x0, d,
                                  alpha_prev, alpha,
                                  phi0, der0, c1, c2, max_ls_iter)

            der_a = self._phi_derivative(g_a, d)

            if abs(der_a) <= -c2 * der0:
                return alpha

            if der_a >= 0:
                return self._zoom(closure, x0, d,
                                  alpha, alpha_prev,
                                  phi0, der0, c1, c2, max_ls_iter)

            alpha_prev = alpha
            phi_prev = phi_a
            alpha = 0.5 * (alpha + alpha_max)

        return alpha

    def step(self, closure):
        
        loss = closure()
        g = self._gather_flat_grad()

        group = self.param_groups[0]
        momentum = group['momentum']
        c1 = group['c1']
        c2 = group['c2']
        alpha1 = group['alpha1']
        alpha_max = group['alpha_max']
        max_ls_iter = group['max_ls_iter']

        g = self._gather_flat_grad()

        # update momentum buffer
        if 'momentum_buffer' not in self.state:
            self.state['momentum_buffer'] = torch.zeros_like(g)
        self.state['momentum_buffer'].mul_(momentum).add_(g, alpha=1)

        d = - self.state['momentum_buffer']

        x0 = self._gather_flat_params()
        phi0 = loss.item()
        der0 = torch.dot(g, d).item()
        alpha = self._strong_wolfe(
            closure, x0, d, phi0, der0,
            c1, c2, alpha1, alpha_max, max_ls_iter
        )
        new_x = x0 + alpha * d
        self._set_params_from_flat(new_x)

        return loss

# Adam optimizer that uses Strong Wolfe line search to determine step length
class Adam_Strong_Wolfe(Optimizer):
    def __init__(self, params, alpha1=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 c1=1e-4, c2=0.9, alpha_max=0.1, max_ls_iter=100):
        
        defaults = dict(alpha1=alpha1, betas=betas, eps=eps,
                        c1=c1, c2=c2, alpha_max=alpha_max,
                        max_ls_iter=max_ls_iter)
        super().__init__(params, defaults)
    
    # recycled from previous optimizer
    def _gather_flat_params(self):
        return torch.cat([p.data.view(-1) for group in self.param_groups for p in group['params']])

    # recycled from previous optimizer
    def _gather_flat_grad(self):
        return torch.cat([p.grad.view(-1) for group in self.param_groups for p in group['params']])

    # recycled from previous optimizer
    def _set_params_from_flat(self, flat):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(flat[idx:idx+numel].view_as(p))
                idx += numel

    # recycled from previous optimizer
    def _phi(self, closure, x0, d, alpha):
        self._set_params_from_flat(x0 + alpha * d)
        loss = closure()
        g = self._gather_flat_grad()
        return loss.item(), g

    # recycled from previous optimizer
    def _phi_derivative(self, grad, d):
        return torch.dot(grad, d).item()

    # zoom phase of Strong Wolfe line search
    def _zoom(self, closure, x0, d,
              alo, ahi, phi0, der0,
              c1, c2, max_ls_iter):

        for _ in range(max_ls_iter):
            aj = 0.5 * (alo + ahi)
            phi_j, g_j = self._phi(closure, x0, d, aj)

            if phi_j > phi0 + c1 * aj * der0:
                ahi = aj
            else:
                der_j = self._phi_derivative(g_j, d)
                if abs(der_j) <= -c2 * der0:
                    return aj
                if der_j * (ahi - alo) >= 0:
                    ahi = alo
                alo = aj

        return aj

    # Strong Wolfe line search
    def _strong_wolfe(self, closure, x0, d,
                      phi0, der0,
                      c1, c2,
                      alpha1, alpha_max,
                      max_ls_iter):

        alpha_prev = 0.0
        phi_prev = phi0

        alpha = alpha1
        for k in range(max_ls_iter):
            phi_a, g_a = self._phi(closure, x0, d, alpha)

            if (phi_a > phi0 + c1 * alpha * der0) or \
               (phi_a >= phi_prev and k > 0):
                return self._zoom(closure, x0, d,
                                  alpha_prev, alpha,
                                  phi0, der0,
                                  c1, c2, max_ls_iter)

            der_a = self._phi_derivative(g_a, d)

            if abs(der_a) <= -c2 * der0:
                return alpha

            if der_a >= 0:
                return self._zoom(closure, x0, d,
                                  alpha, alpha_prev,
                                  phi0, der0,
                                  c1, c2, max_ls_iter)

            alpha_prev = alpha
            phi_prev = phi_a
            alpha = 0.5 * (alpha + alpha_max)

        return alpha

    # step function that implements Adam with Strong Wolfe line search
    def step(self, closure):

        loss = closure()
        g = self._gather_flat_grad()

        group = self.param_groups[0]
        alpha1 = group['alpha1']
        beta1, beta2 = group['betas']
        eps = group['eps']
        c1 = group['c1']
        c2 = group['c2']
        alpha_max = group['alpha_max']
        max_ls_iter = group['max_ls_iter']

        state = self.state
        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(g)     # m_t
            state['exp_avg_sq'] = torch.zeros_like(g)  # v_t
        state['step'] += 1
        step = state['step']
        m = state['exp_avg']
        v = state['exp_avg_sq']

        # Adam updates
        m.mul_(beta1).add_(g, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

        # bias correction
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)

        # search direction
        d = - alpha1 * m_hat / (torch.sqrt(v_hat) + eps)

        x0 = self._gather_flat_params()
        phi0 = loss.item()
        der0 = torch.dot(g, d).item()

        alpha = self._strong_wolfe(
            closure, x0, d, phi0, der0,
            c1, c2, alpha1, alpha_max, max_ls_iter
        )

        # update parameters
        new_x = x0 + alpha * d
        self._set_params_from_flat(new_x)

        return loss

# Trust region method with Cauchy point algorithm
class TrustRegionCauchy(Optimizer):
    def __init__(self, params, r0=1, rmax=5, eta=0.15):
        defaults = dict(r0=r0, rmax=rmax, eta=eta)
        super().__init__(params, defaults)

    # flatten parameter vector
    def _gather_flat_params(self):
        return torch.cat([p.data.view(-1) for group in self.param_groups
                                          for p in group['params']])

    # flatten gradient vector
    def _gather_flat_grad(self):
        return torch.cat([p.grad.view(-1) for group in self.param_groups
                                           for p in group['params']])

    # set parameters from a flat vector
    def _set_params_from_flat(self, flat):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                n = p.numel()
                p.data.copy_(flat[idx:idx+n].view_as(p))
                idx += n

    # Cauchy point algorithm to compute step
    def _cauchy_step(self, g, Hv_g, r):
        g_norm = torch.linalg.norm(g)

        if g_norm < 1e-12:
            return torch.zeros_like(g)

        p_s = -r * g / g_norm
        curvature = torch.dot(g, Hv_g)

        # compute tau
        if curvature <= 0:
            tau = 1.0
        else:
            tau = min((g_norm**3) / (r * curvature), 1.0)

        return tau * p_s

    # model function m_k(p) for model reduction
    def _model_value(self, loss_k, g, p, Hv_p):
        return loss_k + torch.dot(g, p) + 0.5 * torch.dot(p, Hv_p)

    def step(self, closure):
        loss = closure(backward=False)
        loss_k = loss.item()
        params = [p for group in self.param_groups for p in group['params']]
        g_list = torch.autograd.grad(loss, params, create_graph=True)
        g = torch.cat([gi.reshape(-1) for gi in g_list])

        group = self.param_groups[0]
        r = group.get("r_current", group["r0"])
        rmax = group["rmax"]
        eta = group["eta"]

        # Hessian-vector product function
        def hvp(v):
            Hv_list = torch.autograd.grad(
                g,     # these are "outputs"                
                params,     # these are "inputs"
                grad_outputs=v,     # vector to be multiplied
                retain_graph=True
            )
            Hv = torch.cat([h.reshape(-1) for h in Hv_list])
            return Hv

        # Cauchy point
        Hv_g = hvp(g)
        p = self._cauchy_step(g, Hv_g, r)

        # find rho
        x0 = self._gather_flat_params()
        self._set_params_from_flat(x0 + p)
        loss_new = closure(backward=False).item()
        true_red = loss_k - loss_new
        model_red = loss_k - self._model_value(loss_k, g, p, hvp(p))
        rho = true_red / (model_red + 1e-12)

        # trust-region update
        if rho < 0.25:
            r *= 0.25
        elif rho > 0.75 and torch.linalg.norm(p) >= r:
            r = min(2 * r, rmax)

        group["r_current"] = r

        # accept or reject
        if rho <= eta:
            self._set_params_from_flat(x0)
            return loss

        return loss

class Levenberg_Marquardt(Optimizer):
    def __init__(self, params, lambda_=10, lw_iter=10):
        defaults = dict(lambda_=lambda_, lw_iter=lw_iter)
        super().__init__(params, defaults)
    
    # obtains a flattened view of all parameters
    def _gather_flat_params(self):
        return torch.cat([p.data.view(-1) for group in self.param_groups for p in group['params']]) 
    
    # obtains a flattened view of all gradients
    def _gather_flat_grad(self):
        return torch.cat([p.grad.view(-1) for group in self.param_groups for p in group['params']])
    
    # sets the parameters of the model from a flattened view
    def _set_params_from_flat(self, flat):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(flat[idx:idx+numel].view_as(p))
                idx += numel

    def step(self, closure):
        loss = closure(backward=False, lm=True)
        params = [p for group in self.param_groups for p in group['params']]
        g_list = torch.autograd.grad(loss, params, create_graph=True)
        g = torch.cat([gi.reshape(-1) for gi in g_list])

        group = self.param_groups[0]
        lambda_ = group.get('lambda_', 10)    # LM parameter
        max_landweber_iter = group.get('lw_iter', 10)

        # Hessian-vector product function
        def H_GN_v(v):
            Hv_list = torch.autograd.grad(
                g, params, grad_outputs=v,
                retain_graph=True, create_graph=False
            )
            return torch.cat([h.reshape(-1) for h in Hv_list])

        def A_mv(v):
            return H_GN_v(v) + lambda_ * v

        # begin landweber workflow here
        d0 = -g.clone()
        d = d0.clone()
        with torch.no_grad():
            # choose tau using approximation
            Ad0 = A_mv(d0)
            est = (Ad0.norm() / (d0.norm() + 1e-12)).item()
            tau = min(1e-3, 1.0 / (est**2 + 1e-12))
    
            # run Landweber iteration
            for t in range(max_landweber_iter):
                Ad = A_mv(d)
                # print(f"iter {t}: ||d||={d.norm().item():.3e}, ||Ad||={Ad.norm().item():.3e}")
                grad_lw = Ad + g            
                d = d - tau * A_mv(grad_lw)
                if torch.isnan(d).any():
                    raise ValueError("NaN encountered in Landweber iterations.")

        # d is now the LM step
        pk = d

        # update parameters
        x0 = self._gather_flat_params()
        new_x = x0 + pk
        self._set_params_from_flat(new_x)

        return loss

        