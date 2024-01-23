import torch
from laplace.curvature.backpack import _cleanup

from torch.autograd import grad


def manual_jacobian(model, x):
    """
    Assumes input shape: Batch size x dim
    :param model:
    :param x:
    :return:
    """

    with torch.no_grad():

        model.zero_grad()
        with torch.enable_grad():
            x = x.requires_grad_(True)
            f = model(x)
            shape = list(f.shape)
            f_flatten = f.flatten()

            jac = []
            for i, f_i in enumerate(f_flatten):
                parameters = model.parameters()
                params = list(parameters)
                df = list(grad(f_i, params, create_graph=True, retain_graph=True, allow_unused=True))
                for i, df_i in enumerate(df):
                    if df_i is None:
                        df_i = torch.zeros_like(params[i])
                    df[i] = df_i.flatten()
                jac_i = torch.cat(df).detach()
                jac.append(jac_i)
            jac = torch.stack(jac)
            param_num = jac.shape[-1]
            shape.append(param_num)
            jac = jac.reshape(shape)
            model.zero_grad()
            _cleanup(model)
            return jac.detach(), f.detach()
