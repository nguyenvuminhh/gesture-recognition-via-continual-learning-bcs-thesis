import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pointnet2_utils import PointNetSetAbstraction
import provider
from tqdm import tqdm
from kan import KANLayer


# 点云数据的GRU + PointNet++ 模型
class GRUPointNet(nn.Module):
    def __init__(self, num_classes,normal_channel=True, use_KAN=False):
        super(GRUPointNet, self).__init__()
        self.init_kwargs = {"num_classes": num_classes, "normal_channel": normal_channel}
        self.use_KAN = use_KAN
        in_channel = 5 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=64, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=16, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)
        # 定义GRU部分
        self.gru = nn.GRU(input_size=1024, hidden_size=128, batch_first=True, bidirectional=True)
        # 定义全连接层
        if self.use_KAN:
            self.fc1 = KANLayer(256, 512)
            self.fc2 = KANLayer(512, 256)
            self.fc3 = KANLayer(256, num_classes)
        else:
            self.fc1 = nn.Linear(256, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz_sequence):
        # xyz_sequence 应该是一个形状为 (B, C, N) 的张量
        # xyz_sequence 应该是一个形状为 (B, C, N) 的张量
        # B 是批量大小，T 是帧数，N 是每帧的点数，C 是输入通道数
        batch_size, _, num_points = xyz_sequence.size()
        num_frames = 28
        point_per_frame = 120 # 记得改
        # 通过PointNet++处理每一帧
        point_features = []
        for t in range(num_frames):
            xyz = xyz_sequence[:, :3, t*point_per_frame:(t+1)*point_per_frame]
            if self.normal_channel:
                norm = xyz_sequence[:, 3:, t*point_per_frame:(t+1)*point_per_frame]
            else:
                norm = None
            # 使用PointNet++提取每一帧的特征
            l1_xyz, l1_points = self.sa1(xyz, norm)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            point_features.append(l3_points)
        # 将特征列表转换为张量
        point_features = torch.stack(point_features, dim=1)
        avg_features = point_features.squeeze(-1)
        # 通过LGRU层前向传播
        gru_out, (hn, cn) = self.gru(avg_features)
        # 只取最后一个时间步的输出
        x = gru_out[:, -1, :]
        # 通过全连接层
        if self.use_KAN:
            x = self.fc1(x)
            x = self.fc2(x[0])
            x = self.fc3(x[0])
            x = x[0]
        else:
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.4, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=0.4, training=self.training)
            x = self.fc3(x)
        return x




class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def run_post_context(self):
        pass
    
    def run_post_iteration(self):
        pass

    def get_base_loss(self, pred, target):
        return F.cross_entropy(pred, target)

    def forward(self, pred, target, **kwargs):
        return self.get_base_loss(pred, target)


class get_ewc_loss(get_loss):
    def __init__(self, model, ewc_lambda, dataloader, nof_active_classes, args):
        super(get_ewc_loss, self).__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.set_dataloader(dataloader)        
        # self.dataloader = dataloader
        self.nof_active_classes = nof_active_classes
        self.args = args

        self.fisher_list = []
        self.params_list = []
    
    def set_dataloader(self, dataloader):
        self.dataloader = torch.utils.data.DataLoader(
            dataloader.dataset, 
            batch_size=64
        )

    def add_class(self, nof_extra_classes):
        self.nof_active_classes += nof_extra_classes
    
    def run_post_context(self):
        self._compute_fisher()

    def _compute_fisher(self):
        # NOTE: Calls after finishing training a task (per context)
        print("Computing Fisher information matrix...")
        self.model.eval()

        fisher = {}
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)
                params[name] = param.detach().clone()

        nof_batch = len(self.dataloader)

        for batch_id, (points, target) in tqdm(enumerate(self.dataloader, 0), total=len(self.dataloader), smoothing=0.9):
            points = provider.data_augmentation(points)
            if not self.args.use_cpu:
                points = points.cuda()
                target = target.cuda()
            
            self.model.train()
            output = self.model(points)
            
            with torch.no_grad():
                label_weights = F.softmax(output, dim=1)
            
            for class_idx in range(self.nof_active_classes):
                class_target = torch.full((output.shape[0],), class_idx, dtype=torch.long)
                if not self.args.use_cpu:
                    class_target = class_target.cuda()
                
                neg_log_likelihood = F.cross_entropy(output, class_target, reduction='mean')
                
                self.model.zero_grad()
                
                retain_graph = (class_idx + 1) < self.nof_active_classes
                neg_log_likelihood.backward(retain_graph=retain_graph)
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        weight = label_weights[:, class_idx].mean().item()
                fisher[name] += weight * (param.grad.detach() ** 2)

        for name in fisher:
            fisher[name] /= nof_batch

        self.fisher_list.append(fisher)
        self.params_list.append(params)

    def forward(self, pred, target, **kwargs):
        loss = self.get_base_loss(pred, target)

        for fisher, params in zip(self.fisher_list, self.params_list):
            for name, param in self.model.named_parameters():
                if name in fisher:
                    loss += 0.5 * self.ewc_lambda * (fisher[name] * (param - params[name])**2).sum()

        return loss

    
class get_si_loss(get_loss):
    def __init__(self, model, si_lambda):
        super().__init__()
        self.model = model
        self.si_lambda = si_lambda
        self.epsilon = 0.1

        self.prev_iter_params = {}
        self.prev_context_params = {}
        self.omega = {}
        self.w = {}

        self._init_si_buffers()

    def _init_si_buffers(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_iter_params[name] = param.detach().clone()
                self.prev_context_params[name] = param.detach().clone()
                self.omega[name] = torch.zeros_like(param)
                self.w[name] = torch.zeros_like(param)
                
    def run_post_context(self):
        self._update_omega()

    def run_post_iteration(self):
        self._update_W()

    def _update_W(self):
        # NOTE: Called every backward step (per iteration)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    delta = param.detach() - self.prev_iter_params[name]
                    self.w[name] -= (param.grad.detach() * delta)
                self.prev_iter_params[name] = param.detach().clone()

    def _update_omega(self):
        # NOTE: Called after finishing training a task (per context)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta = param.detach() - self.prev_context_params[name]
                self.omega[name] += self.w[name] / (delta ** 2 + self.epsilon)
                self.w[name] = torch.zeros_like(param)
                self.prev_context_params[name] = param.detach().clone()

    def forward(self, pred, target, **kwargs):
        loss = self.get_base_loss(pred, target)

        si_penalty = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta = param - self.prev_context_params[name]
                si_penalty += (self.omega[name] * delta ** 2).sum()

        return loss + self.si_lambda * si_penalty
    
class get_lwf_loss(get_loss):
    def __init__(self, model, lwf_lambda, temperature):
        super().__init__()
        self.model = model
        self.lwf_lambda = lwf_lambda
        self.T = temperature
        self.old_model = None
        self.current_context_number = 1

    def run_post_context(self):
        self.update_old_model()
        self.current_context_number += 1

    def update_old_model(self):
        # NOTE: Called after finishing training a task (per context)
        self.old_model = self._copy_model()

    def _copy_model(self):
        model_copy = type(self.model)(**self.model.init_kwargs)
        model_copy.load_state_dict(self.model.state_dict())
        model_copy.eval()
        model_copy = model_copy.to(next(self.model.parameters()).device)
        for p in model_copy.parameters():
            p.requires_grad = False
        return model_copy

    def forward(self, pred, target, inputs, **kwargs):
        loss = self.get_base_loss(pred, target)
        if self.current_context_number == 1:
            return loss
        lwf_loss = 0
        with torch.no_grad():
            old_logits = self.old_model(inputs)
            p_old = F.softmax(old_logits / self.T, dim=1)

        p_current = F.softmax(pred / self.T, dim=1)

        lwf_loss += - (p_old * torch.log(p_current + 1e-8)).sum(dim=1).mean()

        loss += self.lwf_lambda * lwf_loss
        return loss

def get_current_gradients(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

def get_replay_gradients(model, replay_dataset, data_augmentation_fn, criterion):
    model.zero_grad()
    replayDataLoader = DataLoader(replay_dataset, batch_size=len(replay_dataset), shuffle=False)

    points, target = next(iter(replayDataLoader))
    points = data_augmentation_fn(points)

    points = points.cuda()
    target = target.cuda()
    
    pred = model(points)
    loss = criterion(pred, target.long(), inputs=points)

    loss.backward()
    gradient = get_current_gradients(model)

    return gradient
    