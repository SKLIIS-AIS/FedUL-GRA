import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet_UL(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=6,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)
        self.classifier_ul = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        a = self.classifier(x)
        b = self.classifier_ul(x)
        return torch.cat((a, b), dim=1)


class LeNet_UL_Independent(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.num_classes = num_classes
        self.unlearn_mode = 'samples'
        self.unlearn_alpha = 0.9
        self.unlearn_beta = 1.0

        # Learning branch. Names match LeNet for strict=False state loading.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)

        # Auxiliary unlearning branch.
        self.conv1_ul = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2_ul = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc_1_ul = nn.Linear(16 * 4 * 4, 120)
        self.fc_2_ul = nn.Linear(120, 84)
        self.classifier_ul = nn.Linear(84, num_classes)
        self.use_hard_agreement_gate = True
        self.use_negative_expert = True
        self.register_buffer('negative_threshold', torch.tensor(1.0e6))
        self.register_buffer('negative_target_class', torch.tensor(-1, dtype=torch.long))
        self.register_buffer('negative_calibrated', torch.tensor(0.0))

    def set_ablation(self, disable_cne=False, disable_gra=False):
        self.use_negative_expert = not disable_cne
        self.use_hard_agreement_gate = not disable_gra

    def set_unlearn_rule(self, mode='samples', alpha=0.9, beta=1.0, target_class=None):
        self.unlearn_mode = mode
        self.unlearn_alpha = alpha
        self.unlearn_beta = beta
        if target_class is not None:
            self.negative_target_class.fill_(int(target_class))

    def _learning_features(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return x

    def _learning_logits(self, x):
        return self.classifier(self._learning_features(x))

    def _auxiliary_logits(self, x):
        return self.classifier_ul(self._auxiliary_features(x))

    def _auxiliary_features(self, x):
        x = self.conv1_ul(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2_ul(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc_1_ul(x))
        x = F.relu(self.fc_2_ul(x))
        return x

    def set_negative_expert(self, threshold=None, target_class=None, calibrated=True):
        if threshold is not None:
            self.negative_threshold.fill_(float(threshold))
        if target_class is not None:
            self.negative_target_class.fill_(int(target_class))
        self.negative_calibrated.fill_(1.0 if calibrated else 0.0)

    def _negative_evidence_from_logits(self, out_l, out_a):
        p_l = torch.softmax(out_l.detach(), dim=1)
        p_a = torch.softmax(out_a.detach(), dim=1)
        _conf_l, pred_l = p_l.max(dim=1)
        conf_a, pred_a = p_a.max(dim=1)
        aux_support = p_a.gather(1, pred_l.unsqueeze(1)).squeeze(1)
        disagree = pred_a.ne(pred_l).float()
        return (conf_a - aux_support).clamp_min(0.0) + 0.5 * disagree

    def negative_evidence(self, x):
        out_l = self._learning_logits(x)
        out_a = self._auxiliary_logits(x)
        return self._negative_evidence_from_logits(out_l, out_a)

    def _override_mask(self, out_l, out_a):
        if not self.use_negative_expert:
            return torch.zeros(out_l.size(0), dtype=torch.bool, device=out_l.device)
        if not self.use_hard_agreement_gate:
            return torch.ones(out_l.size(0), dtype=torch.bool, device=out_l.device)
        if self.unlearn_mode == 'class':
            return torch.ones(out_l.size(0), dtype=torch.bool, device=out_l.device)
        evidence = self._negative_evidence_from_logits(out_l, out_a)
        mask = evidence >= self.negative_threshold.to(out_l.device)
        target_class = int(self.negative_target_class.item())
        if target_class >= 0:
            pred_l = out_l.detach().argmax(dim=1)
            mask = mask & pred_l.eq(target_class)
        return mask

    def _negative_override(self, out_l, out_a):
        logits = out_l.clone()
        mask = self._override_mask(out_l, out_a)
        if not mask.any():
            return logits

        target_class = int(self.negative_target_class.item())
        if self.unlearn_mode == 'class' and target_class < 0:
            return logits

        if target_class >= 0:
            logits[mask, target_class] = -1.0e6
        else:
            pred_l = out_l.detach().argmax(dim=1)
            rows = torch.arange(out_l.size(0), device=out_l.device)[mask]
            logits[rows, pred_l[mask]] = -1.0e6
        return logits

    def _retain_route(self, out_l, out_a):
        if not self.use_negative_expert:
            return torch.ones(out_l.size(0), 1, dtype=out_l.dtype, device=out_l.device)
        if not self.use_hard_agreement_gate:
            return torch.zeros(out_l.size(0), 1, dtype=out_l.dtype, device=out_l.device)
        mask = self._override_mask(out_l, out_a)
        return (~mask).to(out_l.dtype).unsqueeze(1)

    def gra_gate(self, x):
        out_l = self._learning_logits(x)
        out_a = self._auxiliary_logits(x)
        return self._retain_route(out_l, out_a)

    def forward(self, x, mode=None):
        mode = mode or ('train' if self.training else 'unlearn')
        feat_l = self._learning_features(x)
        out_l = self.classifier(feat_l)
        feat_a = self._auxiliary_features(x)
        out_a = self.classifier_ul(feat_a)

        if mode == 'train':
            return torch.cat((out_l, out_a), dim=1)
        if mode == 'unlearn':
            override_logits = self._negative_override(out_l, out_a)
            route = self._retain_route(out_l, out_a)
            return route * out_l + (1.0 - route) * override_logits

        raise ValueError('Unknown LeNet_UL_Independent mode: {}'.format(mode))


def lenet_ul(**kwargs):
    model = LeNet_UL(**kwargs)
    return model


def lenet_ul_independent(**kwargs):
    model = LeNet_UL_Independent(**kwargs)
    return model
