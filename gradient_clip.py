import torch

def adjust_gradient(model, optim, accelerator, norm_grad, loss_a, loss_b, lambda_=1):
    # Clear gradients
    optim.zero_grad()

    # Calculate gradients for loss_b
    accelerator.backward(loss_b, retain_graph=True)
    norm_grad()
    b_grads = [p[1].grad.clone() for p in model.named_parameters() if ("attn2" in p[0] and p[1].grad != None)]
    # Clear gradients
    optim.zero_grad()

    # Calculate gradients for loss_a
    # loss_a.backward()
    accelerator.backward(loss_a)
    norm_grad()

    # Gradient adjustment
    # Iterate through model parameters and adjust gradients
    for (p, b_grad) in zip([p[1] for p in model.named_parameters() if ("attn2" in p[0] and p[1].grad != None)], b_grads):
        if p.grad is not None and b_grad is not None:
            # Normalize gradients
            b_grad_norm = b_grad / (torch.linalg.norm(b_grad) + 1e-8)
            a_grad_norm = p.grad / (torch.linalg.norm(p.grad) + 1e-8)
            # Calculate dot product between gradients
            dot_product = torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten())
            # If gradients are in opposite directions, adjust gradient
            if dot_product < 0:
                adjustment = lambda_ * dot_product * b_grad_norm
                p.grad -= adjustment

    # Apply gradient updates
    optim.step()
    optim.zero_grad()