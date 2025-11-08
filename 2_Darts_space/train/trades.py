import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.cuda.amp import autocast



def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_logits, _ = model(x_adv)
                clean_logits, _ = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(adv_logits, dim=1), #model(x_adv)
                                       F.softmax(clean_logits, dim=1)) #model(x_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    clean_logits_new, _ = model(x_natural)
    adv_logits_new, _ = model(x_natural)
    loss_natural = F.cross_entropy(clean_logits_new, y) #model(x_natural)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits_new, dim=1), #model(x_adv)
                                                    F.softmax(clean_logits_new, dim=1)) #model(x_natural)
    loss = loss_natural + beta * loss_robust
    return loss


def madry_loss(model,
               x_natural,
               y,
               optimizer,
               step_size=0.003,
               epsilon=0.031,
               perturb_steps=10,
               distance='l_inf',
               ):
    # define KL-loss
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits, _ = model(x_adv)
                loss_ce = criterion_ce(logits, y).mean()
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits, _ = model(x_adv)
    loss = F.cross_entropy(logits, y)

    return loss

def mart_loss(model, x_natural, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=6.0,  attack='linf-pgd'):
    """
    MART training (Wang et al, 2020).
    """
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits, _ = model(x_adv)
                loss_ce = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        raise ValueError(f'Attack={attack} not supported for MART training!')
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits,_ = model(x_natural)
    logits_adv,_ = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    nat_probs = F.softmax(logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss


def mart_loss_safe(model, x_natural, y, optimizer,
                   step_size=0.007, epsilon=0.031,
                   perturb_steps=7, beta=6.0,
                   epoch=0, warmup_epochs=8, eps=1e-6):
    """
    稳定版 MART 损失函数：
    - 自动 α 软切换（从CE平滑过渡到 MART）
    - 自动调节 beta、攻击强度
    - 防止 KL / log(1-p) 爆炸
    """
    kl = torch.nn.KLDivLoss(reduction='none')
    model.eval()

    # α：当前 epoch 的软切换系数（0→1）
    alpha = min(1.0, epoch / warmup_epochs)
    beta_eff = 0.5 + (beta - 0.5) * alpha
    steps_eff = int(1 + (perturb_steps - 1) * alpha)
    step_size_eff = 0.003 + (step_size - 0.003) * alpha
    epsilon_eff = 0.02 + (epsilon - 0.02) * alpha
    temperature = 1.0 + (2.0 - 1.0) * (1.0 - alpha)

    # 生成对抗样本
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
    for _ in range(steps_eff):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits_adv, _ = model(x_adv)
            loss_ce = F.cross_entropy(logits_adv, y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size_eff * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon_eff), x_natural + epsilon_eff)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    optimizer.zero_grad()

    logits_nat, _ = model(x_natural)
    logits_adv, _ = model(x_adv)

    adv_probs = F.softmax(logits_adv / temperature, dim=1)
    nat_probs = F.softmax(logits_nat / temperature, dim=1)
    adv_probs = torch.clamp(adv_probs, eps, 1.0 - eps)
    nat_probs = torch.clamp(nat_probs, eps, 1.0 - eps)

    # 构造次优标签
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    # 各项损失计算
    loss_ce_nat = F.cross_entropy(logits_nat, y)
    loss_adv = F.cross_entropy(logits_adv, y)
    loss_aux = F.nll_loss(torch.log(1.0 - adv_probs + eps), new_y)

    # 计算 true_probs 及其 mask
    true_probs = torch.gather(nat_probs, 1, y.unsqueeze(1)).squeeze()
    mask = (true_probs > 0.1).detach()

    # 稳定的 KL 项（mask 保护 + 权重截断）
    if mask.sum() > 0:
        kl_term = kl(torch.log(adv_probs[mask]), nat_probs[mask])
        weight = 1.0 - true_probs[mask]
        weight = torch.clamp(weight, 0.0, 0.5 + 0.5 * alpha)
        loss_robust = torch.sum(torch.sum(kl_term, dim=1) * weight) / x_natural.size(0)
    else:
        loss_robust = torch.tensor(0.0).to(x_natural.device)

    loss_mart = loss_adv + loss_aux + beta_eff * loss_robust
    loss = (1 - alpha) * loss_ce_nat + alpha * loss_mart

    return loss


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.nn as nn

def mart_loss_safe_amp(model, x_natural, y,
                       step_size=0.007, epsilon=0.031,
                       perturb_steps=7, beta=6.0,
                       epoch=0, warmup_epochs=8, eps=1e-6):
    """
    AMP 混合精度版 MART 损失函数：
    - 攻击阶段 (PGD) 使用 FP32，防止梯度失真。
    - 主计算阶段使用 autocast() 半精度。
    - 数值稳定处理，避免 log/softmax 下溢。
    """
    kl = nn.KLDivLoss(reduction='none')
    device = x_natural.device

    alpha = min(1.0, epoch / warmup_epochs)
    beta_eff = 0.5 + (beta - 0.5) * alpha
    steps_eff = int(1 + (perturb_steps - 1) * alpha)
    step_size_eff = 0.003 + (step_size - 0.003) * alpha
    epsilon_eff = 0.02 + (epsilon - 0.02) * alpha
    temperature = 1.0 + (2.0 - 1.0) * (1.0 - alpha)

    # FP32攻击生成
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
    for _ in range(steps_eff):
        x_adv.requires_grad_()
        with torch.cuda.amp.autocast(enabled=False):
            logits_adv, _ = model(x_adv.float())
            loss_ce = F.cross_entropy(logits_adv, y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size_eff * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon_eff), x_natural + epsilon_eff)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    with autocast():
        logits_nat, _ = model(x_natural)
        logits_adv, _ = model(x_adv)

        adv_probs = F.softmax(logits_adv / temperature, dim=1)
        nat_probs = F.softmax(logits_nat / temperature, dim=1)
        adv_probs = torch.clamp(adv_probs, eps, 1.0 - eps)
        nat_probs = torch.clamp(nat_probs, eps, 1.0 - eps)

        tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

        loss_ce_nat = F.cross_entropy(logits_nat, y)
        loss_adv = F.cross_entropy(logits_adv, y)
        loss_aux = F.nll_loss(torch.log(1.0 - adv_probs + eps), new_y)

        true_probs = torch.gather(nat_probs, 1, y.unsqueeze(1)).squeeze()
        mask = (true_probs > 0.1).detach()

        if mask.sum() > 0:
            kl_term = kl(torch.log(adv_probs[mask]), nat_probs[mask])
            weight = 1.0 - true_probs[mask]
            weight = torch.clamp(weight, 0.0, 0.5 + 0.5 * alpha)
            loss_robust = torch.sum(torch.sum(kl_term, dim=1) * weight) / x_natural.size(0)
        else:
            loss_robust = torch.tensor(0.0, device=device, dtype=logits_nat.dtype)

        loss_mart = loss_adv + loss_aux + beta_eff * loss_robust
        loss_total = (1 - alpha) * loss_ce_nat + alpha * loss_mart

    return loss_total


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


def mart_loss_safe_light(model, x_natural, y, optimizer,
                         step_size=0.007, epsilon=0.031,
                         perturb_steps=7, beta=6.0,
                         epoch=0, warmup_epochs=8, eps=1e-6):
    """
    显存优化版 MART_SAFE：
    - 攻击阶段 no_grad + 手动 retain_grad 控制；
    - 防止计算图累积；
    - 对 KL 和 mask 部分 detach；
    - 全精度，无 AMP。
    """

    # ========== warmup & 动态参数 ==========
    kl = nn.KLDivLoss(reduction='none')
    alpha = min(1.0, epoch / warmup_epochs)
    beta_eff = 0.5 + (beta - 0.5) * alpha
    steps_eff = int(1 + (perturb_steps - 1) * alpha)
    step_size_eff = 0.003 + (step_size - 0.003) * alpha
    epsilon_eff = 0.02 + (epsilon - 0.02) * alpha
    temperature = 1.0 + (2.0 - 1.0) * (1.0 - alpha)

    # ========== 对抗样本生成阶段 (无梯度累积) ==========
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
    for _ in range(steps_eff):
        x_adv.requires_grad_()
        with torch.enable_grad():  # 仅此一层启用梯度
            logits_adv, _ = model(x_adv)
            loss_ce = F.cross_entropy(logits_adv, y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size_eff * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon_eff), x_natural + epsilon_eff)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        del grad, loss_ce, logits_adv  #
        torch.cuda.empty_cache()
    x_adv = x_adv.detach()  # 彻底断开计算图

    # ========== 计算损失阶段 (train 模式) ==========
    model.train()
    optimizer.zero_grad()

    logits_nat, _ = model(x_natural)
    logits_adv, _ = model(x_adv)

    # softmax 结果 detach 非必要路径
    adv_probs = F.softmax(logits_adv / temperature, dim=1)
    nat_probs = F.softmax(logits_nat / temperature, dim=1)
    adv_probs = torch.clamp(adv_probs, eps, 1.0 - eps)
    nat_probs = torch.clamp(nat_probs, eps, 1.0 - eps)

    # 构造次优标签
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    # 各项损失
    loss_ce_nat = F.cross_entropy(logits_nat, y)
    loss_adv = F.cross_entropy(logits_adv, y)
    loss_aux = F.nll_loss(torch.log(1.0 - adv_probs + eps), new_y)

    # 计算 true_probs 并 detach
    true_probs = torch.gather(nat_probs.detach(), 1, y.unsqueeze(1)).squeeze()
    mask = (true_probs > 0.1).detach()

    # KL 稳定项
    if mask.sum() > 0:
        # 仅对有效样本计算 KL
        kl_term = kl(torch.log(adv_probs[mask]), nat_probs[mask].detach())
        weight = (1.0 - true_probs[mask]).clamp(0.0, 0.5 + 0.5 * alpha)
        loss_robust = torch.sum(torch.sum(kl_term, dim=1) * weight) / x_natural.size(0)
    else:
        loss_robust = torch.tensor(0.0, device=x_natural.device)

    loss_mart = loss_adv + loss_aux + beta_eff * loss_robust
    loss_total = (1 - alpha) * loss_ce_nat + alpha * loss_mart

    # 显存释放
    del logits_nat, logits_adv, adv_probs, nat_probs, kl_term
    torch.cuda.empty_cache()

    return loss_total
