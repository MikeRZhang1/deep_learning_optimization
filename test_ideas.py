import torch
from torch.optim import Optimizer

class MomentumSGD_StrongWolfe(Optimizer):
    """
    Momentum SGD with Strong Wolfe line search.
    Designed for full-batch optimization (deterministic gradients).
    """

    def __init__(self, params, lr=1.0, momentum=0.9, c1=1e-4, c2=0.9, 
                 alpha_max=10.0, max_ls_iter=20):
        defaults = dict(lr=lr, momentum=momentum, c1=c1, c2=c2,
                        alpha_max=alpha_max, max_ls_iter=max_ls_iter)
        super().__init__(params, defaults)

    # -------------------------------------------------------------
    # Utility: flatten parameters and gradients
    # -------------------------------------------------------------
    def _gather_flat_params(self):
        return torch.cat([p.data.view(-1) for group in self.param_groups
                                         for p in group['params']])

    def _gather_flat_grad(self):
        return torch.cat([p.grad.view(-1) for group in self.param_groups
                                          for p in group['params']])

    def _set_params_from_flat(self, flat):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(flat[idx:idx+numel].view_as(p))
                idx += numel

    # -------------------------------------------------------------
    # φ(α) = f(x + α d);  Dφ(α) = ∇f(x+α d)^T d
    # -------------------------------------------------------------
    def _phi(self, closure, x0, d, alpha):
        self._set_params_from_flat(x0 + alpha * d)
        loss = closure()
        g = self._gather_flat_grad()
        return loss.item(), g

    def _phi_derivative(self, grad, d):
        return torch.dot(grad, d).item()

    # -------------------------------------------------------------
    # Strong Wolfe line search: zoom subroutine
    # -------------------------------------------------------------
    def _zoom(self, closure, x0, d, alo, ahi, phi0, der0, c1, c2):
        for _ in range(20):
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

    # -------------------------------------------------------------
    # Strong Wolfe line search (main routine)
    # -------------------------------------------------------------
    def _strong_wolfe(self, closure, x0, d, phi0, der0, 
                      c1, c2, alpha1, alpha_max):
        alpha_prev = 0.0
        phi_prev = phi0

        alpha = alpha1
        for _ in range(20):
            phi_a, g_a = self._phi(closure, x0, d, alpha)

            # Armijo violation or non-decrease
            if (phi_a > phi0 + c1 * alpha * der0) or \
               (phi_a >= phi_prev and _ > 0):
                return self._zoom(closure, x0, d,
                                  alpha_prev, alpha,
                                  phi0, der0, c1, c2)

            der_a = self._phi_derivative(g_a, d)

            # Strong Wolfe curvature condition satisfied
            if abs(der_a) <= -c2 * der0:
                return alpha

            # Derivative becomes positive → zoom
            if der_a >= 0:
                return self._zoom(closure, x0, d,
                                  alpha, alpha_prev,
                                  phi0, der0, c1, c2)

            # Otherwise expand the interval
            alpha_prev = alpha
            phi_prev = phi_a
            alpha = 0.5 * (alpha + alpha_max)

        return alpha

    # -------------------------------------------------------------
    # STEP: Momentum + line search
    # -------------------------------------------------------------
    def step(self, closure):
        """
        closure() should:
           - recompute forward pass
           - return loss
           - compute gradients
        """
        loss = closure()

        # Collect gradients
        g = self._gather_flat_grad()

        # Initialize momentum buffer and direction
        for group in self.param_groups:
            momentum = group['momentum']
            c1, c2 = group['c1'], group['c2']
            alpha_max = group['alpha_max']

        if 'momentum_buffer' not in self.state:
            self.state['momentum_buffer'] = g.clone().neg()
        else:
            buf = self.state['momentum_buffer']
            buf.mul_(momentum).add_(g, alpha=-1.0)
        d = self.state['momentum_buffer']     # search direction

        # Flatten parameters
        x0 = self._gather_flat_params()

        # Compute φ(0) and Dφ(0)
        phi0 = loss.item()
        der0 = torch.dot(g, d).item()

        # Initial guess for α
        alpha1 = 1.0

        # Perform strong Wolfe search
        alpha = self._strong_wolfe(
            closure, x0, d, phi0, der0,
            c1, c2, alpha1, alpha_max
        )

        # Update parameters
        new_x = x0 + alpha * d
        self._set_params_from_flat(new_x)

        return loss
