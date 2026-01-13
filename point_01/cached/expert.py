class TemporalMoE(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.expert1 = nn.Conv1d(in_dim, in_dim, 3, padding=1, groups=in_dim) # 常规卷积
        self.expert2 = nn.Conv1d(in_dim, in_dim, 3, padding=2, dilation=2, groups=in_dim) # 空洞卷积
        self.expert3 = nn.AvgPool1d(3, stride=1, padding=1) # 平滑专家
        self.gate = nn.Sequential( # 门控网络
            nn.Linear(in_dim, 3),
            nn.Softmax(dim=-1)
        )
    def forward(self, x): # x: (B, T, C)
        B, T, C = x.shape
        x_t = x.transpose(1, 2) # (B, C, T)
        w = self.gate(x.mean(dim=1)) # (B, 3) 根据全局特征产生权重
        out = w[:, 0:1, None] * self.expert1(x_t) + \
              w[:, 1:2, None] * self.expert2(x_t) + \
              w[:, 2:3, None] * self.expert3(x_t)
        return out.transpose(1, 2)