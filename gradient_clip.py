import torch

def adjust_gradient(model, optim, accelerator, norm_grad, loss_a, loss_b, lambda_=1):
    optim.zero_grad()

    accelerator.backward(loss_b, retain_graph=True)
    b_grads = [p[1].grad.clone() for p in model.named_parameters() if (p[1].grad != None)]

    optim.zero_grad()
    accelerator.backward(loss_a, retain_graph=True)
    
    total_params = 0
    adj_params = 0
    for (p, b_grad) in zip([p[1] for p in model.named_parameters() if (p[1].grad != None)], b_grads):
        if p.grad is not None and b_grad is not None:
            len_b = torch.linalg.norm(b_grad)
            b_grad_norm = b_grad.clone()
            a_grad_norm = p.grad.clone()
            dot_product = torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) 
            if dot_product < 0:
                adjustment = lambda_ * dot_product / (len_b *len_b) * b_grad
                p.grad -= adjustment
                adj_params += len(adjustment.flatten())
            total_params += len(p.grad.flatten())

    optim.step()
    optim.zero_grad()
    return adj_params, total_params