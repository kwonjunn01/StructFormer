import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import json

NEG_INF = -1000000

class Mlp(nn.Module):
    r"""2-layer MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DynamicPosBias(nn.Module):
    r"""DPB module
    
    Use a MLP to predict position bias used in attention.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        # self.noise2  =noise2 = nn.Parameter()
        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            
            # generate mother-set
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2Ww-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases, persistent=False)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index, persistent=False)
        self.noise2 = nn.Parameter(torch.randn(3,256)).cuda()*0.1
        self.noise3 = nn.Parameter(torch.randn(3,512)).cuda()*0.1
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, token,clk, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        B , K_,C = token.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        t_qkv = self.qkv(token).reshape(B, K_, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        _, _ ,qtoken= t_qkv[0], t_qkv[1], t_qkv[2]
        
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # aux, token, kl_loss, aux_attn = self.structlayer(k.reshape(B, -1, self.num_heads, C // self.num_heads), v.permute(0,2,1,3).reshape(B, -1, self.num_heads, C // self.num_heads), token,clk,relative_position_bias.unsqueeze(0))
        # @ stands for matrix multiplication
        attn = (q @ k.transpose(-2, -1))
        # attn = ((q+aux.reshape(B_,self.num_heads,N,C // self.num_heads)) @ (k+aux.reshape(B_,self.num_heads,N,C // self.num_heads)).transpose(-2, -1))
        if self.position_bias:
            pos = self.pos(self.biases) # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # aux, token, kl_loss, aux_attn = self.structlayer(x.reshape(B, -1, self.num_heads, C // self.num_heads), x.reshape(B, -1, self.num_heads, C // self.num_heads), token,clk,relative_position_bias.unsqueeze(0))
        aux, token, kl_loss, aux_attn = self.structlayer(x.reshape(B, -1, self.num_heads, C // self.num_heads), x.reshape(B, -1, self.num_heads, C // self.num_heads), qtoken,clk,relative_position_bias.unsqueeze(0))
        x = self.proj(x+ aux.reshape(B_, N, C)*0.5)
        token = self.proj(token)
        x = self.proj_drop(x)
        
        # aux = self.proj(aux.reshape(B_, N, C))
        return x, token, kl_loss

    def extra_repr(self) -> str:
        return f'dim={self.dim}, group_size={self.group_size}, num_heads={self.num_heads}'
    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.
        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.
        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.
        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
    def structlayer(self, k, v, token,clk,bias):
        B, N, Ch,Cs = k.shape
        B, K_,C = token.shape

        if clk >= 2:
            noise = self.noise2.expand(B,-1,-1)

            if clk == 3:
                noise = self.noise3.expand(B,-1,-1)
                cycle = 1
            else: 
                cycle = 1

 
            parent_token = self.EM(k.reshape(B, N, C), v.reshape(B, N, C), token, cycle=cycle, alpha=10).reshape(B, K_, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)* self.scale
            parent_attn = (parent_token @ v.permute(0,2,3,1)) * self.scale
            parent_attn = self.softmax(parent_attn)
            attn = None
            _, p_mask = torch.max(parent_attn, dim=-2) # B 3 N 
            p_std = self.std_cal(v.permute(0,2,1,3), parent_attn, parent_token, p_mask)
            # noise = 0.01*torch.randn_like(token)  # (B,K,C)
            # noise = 0.1*torch.randn_like(token)  # (B,K,C)
            child = torch.repeat_interleave(parent_token.permute(0, 2, 1, 3).reshape(B, K_, C), repeats=2, dim=1)
            paired_noise = torch.zeros_like(child).cuda()
            paired_noise[:, ::2] = p_std * noise
            paired_noise[:, 1::2] = -p_std * noise
            new_token = child + paired_noise
            _, K, _ = new_token.shape
            # semantic_token = new_token.reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)* self.scale
            semantic_token = self.EM(v.reshape(B, N, C),v.reshape(B, N, C), new_token, cycle=1, alpha=10).reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)* self.scale
            sem_attn = (semantic_token @ v.permute(0,2,3,1)) * self.scale # B 3 K N 
            sem_attn = self.softmax(sem_attn)
            _, sem_mask = torch.max(sem_attn, dim=-2) # B 3 N 
            self.semantic_mask = sem_mask.detach().cpu()
            self.p_mask = p_mask.detach().cpu()
            if clk == 3:
                G = int(np.sqrt(N))
                v_reshaped = v.reshape(B, N, C).permute(0, 2, 1).reshape(B, C, G, G)

                # 2. Bilinear interpolation을 사용해 14×14로 업샘플 (원하는 해상도로 조정 가능)
                v_upsampled = F.interpolate(v_reshaped, size=(2*G, 2*G), mode='bilinear', align_corners=False)

                # 3. 다시 flatten: [B, C, 14*14] → [B, 196, C]
                v_new = v_upsampled.reshape(B, C, 4*N).permute(0, 2, 1).reshape(B, -1, Ch, Cs)

                # 이제 v_new (shape: [B, 196, C])를 사용하여 p_mask, semantic_mask를 계산합니다.
                # 예시: 기존 코드에서 v 대신 v_new를 사용하면 됩니다.
                # parent_token_ = self.EM(v_new.reshape(B, -1, C), token, cycle=cycle, alpha=5.0).reshape(B, K_, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) * self.scale
                parent_attn_ = (parent_token @ v_new.permute(0,2,3,1)) * self.scale
                parent_attn_ = self.softmax(parent_attn_)
                attn = None
                _, p_mask_ = torch.max(parent_attn_, dim=-2) # B 3 N 

                # semantic_token_ = self.EM(v_new.reshape(B, -1, C), new_token, cycle=cycle, alpha=5.0).reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)* self.scale
                sem_attn_ = (semantic_token @ v_new.permute(0,2,3,1)) * self.scale # B 3 K N 
                sem_attn_ = self.softmax(sem_attn_)
                _, sem_mask_ = torch.max(sem_attn_, dim=-2) # B 3 N 
                self.semantic_mask_ = sem_mask_.detach().cpu()
                self.p_mask_ = p_mask_.detach().cpu()





            
            child = semantic_token.gather(2, sem_mask.unsqueeze(-1).expand(-1, -1, -1, C // self.num_heads)).transpose(1, 2).reshape(B, N, C)
            parent = parent_token.gather(2, p_mask.unsqueeze(-1).expand(-1, -1, -1, C // self.num_heads)).transpose(1, 2).reshape(B, N, C) 
            kl_loss = 0
            ## Token hierarchy , k = (B, H, N, C)
            if self.training:
                p_std_ = torch.repeat_interleave(p_std, repeats=2, dim=1)
                c_std = self.std_cal(v.permute(0,2,1,3), sem_attn,semantic_token, sem_mask)
                # var_loss1 = F.relu(c_std - p_std_).mean()
                # var_loss2 = self.variance_constraint(c_std, p_std)
                # (B, K_, C) 형태
                kl_loss = self.gauss_cse_loss_vectorized_with_margin(parent_token.transpose(1, 2).reshape(B, K_, C), p_std, semantic_token.transpose(1, 2).reshape(B, K, C), c_std).mean()
            

            x_cen = F.avg_pool1d(v.reshape(B, N, C).transpose(1, 2), N).transpose(1, 2).expand(-1, N, -1)
            c_inner = torch.einsum('bnd,bnd->bn',x_cen, parent).unsqueeze(-1)  # x와 x_aux의 내적
            c_x_norm = torch.einsum('bnd,bnd->bn',x_cen, x_cen).unsqueeze(-1)   # x의 노름의 제곱

            # Tangent Space에 x_aux 투영 (x의 방향 성분 제거)
            c_x_aux_proj= parent - (c_inner / (c_x_norm + 1e-6)) * x_cen
            
            # c_x_aux = self.update_c_x_aux(parent, c_x_aux_proj,x_cen,parent_token.permute(0, 2, 1, 3).reshape(B,K_,C),p_mask,h=1)
            c_x_aux = self.update_c_x_aux_with_clamp(parent, c_x_aux_proj,x_cen, parent_token.permute(0, 2, 1, 3).reshape(B,K_,C),p_mask, 1, 2)

            token = parent_token.permute(0, 2, 1, 3).reshape(B,K_,C)
            inner_product = torch.einsum('bnd,bnd->bn',parent, child).unsqueeze(-1)  # x와 x_aux의 내적
            x_norm_sq = torch.einsum('bnd,bnd->bn',parent, parent).unsqueeze(-1)   # x의 노름의 제곱

            # Tangent Space에 x_aux 투영 (x의 방향 성분 제거)
            x_aux_proj = child- (inner_product / (x_norm_sq + 1e-6)) * parent
            x_aux_proj = x_aux_proj+ c_x_aux
            # if torch.isnan(x_aux_proj.sum()):
            #     x_aux_proj = 0

        elif clk == 1: 
            new_token = token
            _, K, C = new_token.shape
            semantic_token = self.EM(k.reshape(B, N, C),v.reshape(B, N, C), new_token, cycle=1,alpha=2.0).reshape(B, K, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # child_mean

            sem_attn = (semantic_token @ v.permute(0,2,3,1)) * self.scale# B 3 K N + 
            attn = None
            sem_attn = sem_attn.softmax(dim=-1)
            
            _, sem_mask = torch.max(sem_attn, dim=-2) # B 3 N 
            self.semantic_mask = sem_mask.detach().cpu()
            self.p_mask = None
            child = semantic_token.gather(2, sem_mask.unsqueeze(-1).expand(-1, -1, -1, C // self.num_heads)).transpose(1, 2).reshape(B, N, C)
            token = semantic_token.permute(0, 2, 1, 3).reshape(B,K,C)
            kl_loss = 0
            if self.training:
                kl_loss = self.push_away_cosine(token)
            x_cen = F.avg_pool1d(v.reshape(B, N, C).transpose(1, 2), N).transpose(1, 2).expand(-1, N, -1)
            inner_product = torch.einsum('bnd,bnd->bn',x_cen, child).unsqueeze(-1)  # x와 x_aux의 내적
            x_norm_sq = torch.einsum('bnd,bnd->bn',x_cen, x_cen).unsqueeze(-1)   # x의 노름의 제곱
            x_aux_proj = v
            
            # if torch.isnan(x_aux_proj.sum()):
            #     x_aux_proj = 0
        else:
            x_aux_proj = v
            self.semantic_mask = None
            self.p_mask = None
            attn = None
            kl_loss = 0 
            # Tangent Space에 x_aux 투영 (x의 방향 성분 제거)
            
        return x_aux_proj, token, kl_loss, attn
    def update_c_x_aux_with_clamp(self,
        parent: torch.Tensor,         # shape (B,N,C)
        c_x_aux_proj: torch.Tensor,   # shape (B,N,C)
        x_cen: torch.Tensor,          # shape (B,N,C)
        parent_token: torch.Tensor,   # shape (B,K_,C)
        p_mask: torch.Tensor,         # shape (B,H,N)
        h: int = 0,
        threshold: float = 2.0
    ):
        """
        1) 가장 가까운 토큰 best_k[b,n] 찾음
        2) assigned_token[b,n] = p_mask[b,h,n]
        3) best_k == assigned_token -> c_x_aux_proj 적용, else parent 유지
        4) clamp ratio => norm(c_x_aux)/norm(parent) <= threshold

        Returns c_x_aux: (B,N,C)
        """
        B, N, C = parent.shape
        _, K_, _= parent_token.shape

        # -- 1) dist -> best_k
        x_cen_exp     = x_cen.unsqueeze(2)      # (B,N,1,C)
        parent_exp    = parent_token.unsqueeze(1) # (B,1,K_,C)
        dist = (x_cen_exp - parent_exp).norm(dim=-1)  # (B,N,K_)
        best_k = dist.argmin(dim=2)  # (B,N)

        # -- 2) 할당된 토큰
        assigned_token = p_mask[:, h, :]  # (B,N)

        condition = (best_k == assigned_token)  # (B,N) bool

        c_x_aux = parent.clone()  # (B,N,C)

        c_x_aux_flat      = c_x_aux.view(-1, C)         # (B*N, C)
        c_x_aux_proj_flat = c_x_aux_proj.view(-1, C)
        cond_flat         = condition.view(-1)          # (B*N)

        # “조건=true” => c_x_aux_proj
        c_x_aux_flat[cond_flat] = c_x_aux_proj_flat[cond_flat]
        c_x_aux = c_x_aux_flat.view(B, N, C)

        # -- 3) clamp ratio
        # norm of c_x_aux vs. parent
        eps = 1e-6
        norm_c_aux   = c_x_aux.norm(dim=-1, keepdim=True)   # (B,N,1)
        norm_parent  = parent.norm(dim=-1, keepdim=True)    # (B,N,1)

        ratio = norm_c_aux / (norm_parent + eps)  # (B,N,1)
        # if ratio>threshold => scale down
        # c_x_aux => parent + (c_x_aux - parent)*(threshold/ratio)
        too_big = (ratio > threshold)  # (B,N,1) bool

        if too_big.any():
            # vectorized
            ratio_clamp = (threshold / (ratio + eps))  # (B,N,1)
            # => c_x_aux = parent + (c_x_aux-parent) * ratio_clamp
            # or simpler => c_x_aux *= ratio_clamp (around parent), but if we want to center around parent:
            c_x_aux_diff   = c_x_aux - parent
            c_x_aux_diff  *= ratio_clamp.clamp_max(1.0)  # ensure <=1
            c_x_aux_new    = parent + c_x_aux_diff

            # only apply for those patch positions where too_big==True
            too_big_flat = too_big.view(-1)
            c_x_aux_flat = c_x_aux.view(-1, C)
            c_x_aux_new_flat = c_x_aux_new.view(-1, C)

            c_x_aux_flat[too_big_flat] = c_x_aux_new_flat[too_big_flat]
            c_x_aux = c_x_aux_flat.view(B,N,C)

        return c_x_aux


    def EM(self, k,v, token, cycle=3, alpha=2.0):
        """
        표준 Soft k-means 방식의 EM 알고리즘 예시.

        Args:
        x     : (B, N, C)  -> 입력 패치(또는 임베딩)
        token : (B, K, C)  -> 초기 부모(클러스터) 토큰
        cycle : int        -> E/M 반복 횟수
        alpha : float      -> dot product에 곱할 scale(온도 역수에 해당)

        Returns:
        (B, K, C) 형태로 EM 수행 후 업데이트된 부모(클러스터) 토큰
        """

        # mu를 (B, K, C) 형태로 두고 시작
        # mu = self._l2norm(token,dim=-1)  # shape: (B, K, C)
        mu = token # shape: (B, K, C)
        
        with torch.no_grad():
            for i in range(cycle):
                

                sim = alpha * torch.bmm(k, mu.transpose(-1, -2))  # (B,N,C)*(B,C,K) => (B,N,K)

                # row-wise softmax(dim=-1) => 패치 n 기준으로 K개 클러스터 확률 분포
                z = self.softmax(sim)  # (B, N, K), sum_k z_{n,k} = 1

                # -------------------------
                # M-step
                # -------------------------
                # weighted_sum = \sum_n [z_{n,k} * x_n], shape: (B,C,K)
                #   => x.transpose(-1, -2) : (B, C, N)
                #   => bmm(...) : (B, C, N)*(B, N, K) => (B, C, K)
                weighted_sum = torch.bmm(v.transpose(-1, -2), z)

                # 각 클러스터 k에 대한 z_{n,k}의 합: sum_z = (B, 1, K)
                sum_z = z.sum(dim=1, keepdim=True)  # N축(=dim=1) 합

                # mu_k = (weighted_sum_k) / (sum_{n} z_{n,k})
                # shape: (B, C, K)
                mu_update = weighted_sum / (sum_z + 1e-6)
                # if torch.isnan(weighted_sum).any() or torch.isnan(sum_z).any():
                #     print("NaN in weighted_sum or sum_z!")
                # 다시 (B, K, C) 형태로 transpose
                mu = mu_update.transpose(-1, -2) # => (B,K,C)
                # mu = self._l2norm(token,dim=-1)
                # mu = token + mu
        # mu = mu + token*0.01
        return mu
    def std_cal(self, k, attn, mean, mask=None):
        """
        k : (B, H, N, C)
        attn : (B, H, K, N) -- 각 토큰(k축)에 대한 patch(n축) 가중치
        mean : (B, H, K, C) -- 토큰별 가중 평균(부모 토큰)
        mask : (B, H, K, N) or None
            1(또는 True)는 유효 패치, 0(또는 False)는 무시하고 싶은 위치
            만약 None이면 모든 위치 사용
        """
        B, H, K, C = mean.shape

        # 1) k^2 계산
        # shape: (B, H, N, C)
        k_squared = k ** 2

        # 2) 마스크가 있으면 attn에 곱해준다.
        #    ex) mask가 0인 위치 -> attn이 0이 됨 -> 해당 패치를 무시
        if mask is not None:
            # mask shape이 attn과 동일하다고 가정 (B,H,K,N).
            # 단, 일부 코드에서는 (B,H,1,N) 형태일 수도 있으니
            # 필요하면 broadcast 될 수 있게 맞춰줘야 함
            attn = attn * mask.unsqueeze(2)  # (B,H,K,N)

        # 3) 가중치가 적용된 k^2의 합: S2 = sum_{n}( attn_{k,n} * k^2_n )
        #    shape: (B, H, K, C)
        #    => 'bhkn,bhnc->bhkc'
        S2 = torch.einsum('bhkn, bhnc->bhkc', attn, k_squared)

        # 4) 가중치 합: sum_w = sum_{n} attn_{k,n}
        #    shape: (B, H, K, 1)
        sum_w = attn.sum(dim=3, keepdim=True)  # (B,H,K,1)

        # 5) E[X^2] = S2 / sum_w
        e_x2 = S2 / (sum_w + 1e-6)  # (B,H,K,C)

        # 6) 분산(variance) = E[X^2] - (mean^2)
        #    (B, H, K, C)
        variance = e_x2 - (mean ** 2)
        variance = variance.clamp(min=1e-6)

        # 7) 표준편차 std
        std = torch.sqrt(variance)  # (B,H,K,C)

        # 8) shape 변환 (B, K, H*C) 등 필요 시
        std = std.transpose(-2, -1).reshape(B, K, H*C)
        return std

    def kl_divergence(self, p_mean, p_std, c_mean, c_std):
        # Convert log_std to std for KL calculation
        # std2 = torch.exp(log_std2)
        # Gaussian KL-Divergence calculation: mean1/log_std1 is child, mean2/log_std2 is parent
        kl = torch.log(p_std / c_std) + (c_std**2 + (p_mean - c_mean)**2) / (2 * p_std**2) - 0.5 

        return 1 / (1 + kl + 1e-6)
    # def GaussSCE(self, p_mean, p_std, c_mean, c_std):
    def push_away_cosine(self, token, margin=0.5):
        """
        token : (B, K, C)
        margin: 코사인 유사도가 margin보다 크면 그 초과분만큼 패널티를 줌.

        반환: scalar (배치 평균 로스)
        """
        B, K, C = token.shape
        if K < 2:
            return token.new_tensor(0.0)  # 토큰이 1개 이하라면 밀어낼 대상이 없음

        # (1) Pairwise cosine similarity: (B, K, K)
        # 방법 A: broadcasting + reshape
        token_i = token.unsqueeze(2)  # (B, K, 1, C)
        token_j = token.unsqueeze(1)  # (B, 1, K, C)

        # 확장해서 (B,K,K,C)
        token_i_big = token_i.expand(-1, -1, K, -1)  # (B, K, K, C)
        token_j_big = token_j.expand(-1, K, -1, -1)  # (B, K, K, C)

        # (B,K,K,C) -> (B*K*K, C)
        flat_i = token_i_big.reshape(-1, C)
        flat_j = token_j_big.reshape(-1, C)

        # torch.nn.functional.cosine_similarity: (N,C) x (N,C) => (N,)
        cos_flat = torch.nn.functional.cosine_similarity(flat_i, flat_j, dim=-1)  # (B*K*K,)
        cos_mat = cos_flat.view(B, K, K)                                          # (B,K,K)

        # 자기 자신(i=j) 제외
        diag_mask = torch.eye(K, device=token.device, dtype=torch.bool).unsqueeze(0)  # (1,K,K)
        cos_mat = torch.where(diag_mask, token.new_tensor(0.0), cos_mat)

        # (2) margin 기반 벌점: cos - margin
        # cos가 margin보다 크면 cos - margin, 아니면 0
        over_margin = torch.nn.functional.relu(cos_mat - margin)  # (B,K,K)

        # (3) 평균
        loss = over_margin.mean()  # 모든 배치, 모든 토큰 쌍에 대해 평균
        return loss   
    
    
    def gauss_cse_loss_vectorized_with_margin(
    self,
    p_mean,      # (B, K, C) : parent means
    p_std,       # (B, K, D) : parent stds
    c_mean,      # (B, 2K, C): child means
    c_std,       # (B, 2K, D): child stds
    temperature=0.07,
    margin=1.0,
    lambda_pp=0.1,
):
        """
        GaussCSE + Parent-Parent Margin Push-Away

        Args:
        p_mean : shape (B, K, C)
        p_std  : shape (B, K, D)
        c_mean : shape (B, 2K, C)
        c_std  : shape (B, 2K, D)
        temperature (float): InfoNCE temp for child->parent & parent->child
        margin (float): minimal distance margin for parent->parent
        lambda_pp (float): weight for parent-parent push-away loss

        Returns:
        total_loss: forward + reverse + lambda_pp * parent-parent push
        """
        B, K, C_ = p_mean.shape
        # 기존 Child->Parent (Forward) / Parent->Child (Reverse)는
        # self.gauss_cse_loss_vectorized 내부에 이미 구현되어 있다고 가정.
        # 또는 아래 코드를 그대로 복사하여 forward_loss와 reverse_loss 부분을 구현해도 됩니다.

        # 1) 먼저 기존 forward_loss + reverse_loss
        # 여기서는 간단히 "base_loss" 라고 부르겠습니다.
        base_loss = self.gauss_cse_loss_vectorized(
            p_mean, p_std,
            c_mean, c_std,
            temperature=temperature
        )
        # base_loss = forward_loss + reverse_loss

        # 2) Parent->Parent Push-Away
        # sim_pp: (B, K, K) = self.kl_divergence( p_mean[b,k], p_std[b,k], p_mean[b,j], p_std[b,j] )
        # shape => (B, K, K)
        p_mean1 = p_mean.unsqueeze(2)  # (B, K, 1, C)
        p_std1  = p_std.unsqueeze(2)   # (B, K, 1, D)
        p_mean2 = p_mean.unsqueeze(1)  # (B, 1, K, C)
        p_std2  = p_std.unsqueeze(1)   # (B, 1, K, D)

        sim_pp = self.kl_divergence(p_mean1, p_std1, p_mean2, p_std2).mean(dim=-1)  # (B, K, K)

        # kl_distance ~ (1/sim - 1).
        # shape => (B, K, K)
        kl_dist = (1.0 / (sim_pp + 1e-9)) - 1.0

        # i=j 자기 자신은 제외
        diag_mask = torch.eye(K, dtype=torch.bool, device=p_mean.device).unsqueeze(0)  # (1,K,K)
        # margin-based push-away: margin - dist
        # => (B, K, K)
        margin_mat = margin - kl_dist
        # i=j는 계산에서 제외 (mask out)
        margin_mat = torch.where(diag_mask, torch.zeros_like(margin_mat), margin_mat)
        # ReLU
        pp_loss_mat = F.relu(margin_mat)  # (B, K, K)

        # 최종 parent-parent loss: 평균
        parent_parent_loss = pp_loss_mat.mean()

        # 3) 최종 total_loss = base_loss + lambda_pp * parent_parent_loss
        total_loss = base_loss + lambda_pp * parent_parent_loss
        return total_loss

    def gauss_cse_loss_vectorized(
        self,
        p_mean,      # (B, K, C) : Means of parent tokens
        p_std,       # (B, K, D) : Stds  of parent tokens
        c_mean,      # (B, 2K, C): Means of child tokens
        c_std,       # (B, 2K, D): Stds  of child tokens
        temperature=0.07
    ):
        """
        Vectorized Gaussian CSE Loss
        -------------------------------------
        Each parent p_k has two children c_(2k), c_(2k+1).
        We compute child->parent (forward) and parent->child (reverse) objectives
        in a single pass over the batch dimension.

        Args:
            p_mean : (B, K, C)
            p_std  : (B, K, D)
            c_mean : (B, 2K, C)
            c_std  : (B, 2K, D)
            temperature (float): InfoNCE temperature

        Returns:
            total_loss (scalar tensor): Summation of forward and reverse contrastive losses.
        """
        B, K, C_ = p_mean.shape
        _, child_count, _ = c_mean.shape
        assert child_count == 2*K, "Expect 2 child tokens per parent."

        # ---------------------------------------------------------------
        # 1) Child->Parent (Forward)
        # ---------------------------------------------------------------
        # For child i in [0..2K-1], parent index = i//2.
        # We want sim_cp[b,i] = sim( c_mean[b,i], p_mean[b,i//2] ).


        # (a) Gather the parent's distribution for each child index in [0..2K-1].
        #     parent_index = [0,0,1,1,2,2,...,K-1,K-1]
        parent_indices = torch.arange(0, K, device=p_mean.device).repeat_interleave(2)  
        # parent_indices: shape (2K,), e.g. [0,0,1,1,2,2,...,K-1,K-1]

        # shape => (B, 2K, C)
        p_sel_mean = p_mean[:, parent_indices, :]
        # shape => (B, 2K, D)
        p_sel_std  = p_std[:, parent_indices, :]

        # (b) Compute child->parent similarity: (B, 2K)
        # kl_divergence() must support input shapes (B, 2K, *) and broadcast them.
        sim_cp = self.kl_divergence(p_sel_mean, p_sel_std, c_mean, c_std)  # => (B, 2K)
        
        # (c) Compute child->child similarity for negatives.
        # We'll create a 3D broadcast:
        # c_mean1: (B, 2K, 1, C)
        # c_mean2: (B, 1, 2K, C)
        c_mean1 = c_mean.unsqueeze(2)  # (B, 2K, 1, C)
        c_mean2 = c_mean.unsqueeze(1)  # (B, 1, 2K, C)
        c_std1  = c_std.unsqueeze(2)   # (B, 2K, 1, D)
        c_std2  = c_std.unsqueeze(1)   # (B, 1, 2K, D)

        # sim_cc: (B, 2K, 2K)
        sim_cc = self.kl_divergence(c_mean1, c_std1, c_mean2, c_std2)  # shape (B, 2K, 2K)

        # (d) Construct the mask for negatives: child i => exclude children with same parent
        # mask[i, j] = True if j//2 != i//2
        # We'll do this in 2D (2K,2K), then broadcast to (B,2K,2K).
        child_idx = torch.arange(2*K, device=c_mean.device)
        # parent_of_i = i//2
        # We'll build a 2D boolean mask with shape (2K, 2K)
        child_parent = child_idx // 2  # shape (2K,)
        # broadcast compare
        same_parent = (child_parent.unsqueeze(1) == child_parent.unsqueeze(0))  # (2K,2K), True if same
        # We want negatives => ~same_parent (except ignoring i=j if you want)
        # But often we exclude i=j for self-comparison. We'll do that too if you like:
        diag = torch.eye(2*K, dtype=torch.bool, device=c_mean.device)
        neg_mask = (~same_parent) & (~diag)  # or just (~same_parent) if ignoring self is not required

        # We'll compute the forward InfoNCE-like loss in a vectorized manner:
        # forward_loss[i] = -log( exp(sim_cp[i]/temp) / (exp(sim_cp[i]/temp) + sum_{j in negs} exp(sim_cc[i,j]/temp)) )
        # We'll do this for each batch, each child i in [0..2K-1].
        
        # ex_plus: (B,2K)
        ex_plus = torch.exp(sim_cp / temperature).mean(dim=-1)
        # ex_minus: we want sum_{j in neg_mask(i,:)} exp(sim_cc[:, i, j]/temperature)
        # We'll do a masked fill and sum along dim=2.
        # shape: (B,2K,2K)
        ex_cc = torch.exp(sim_cc / temperature).mean(dim=-1)
        # set non-negatives to 0 using a mask
        # We'll create a (2K,2K) -> broadcast to (B,2K,2K).
        # The masked positions are True => keep, False => zero out
        ex_cc_neg = ex_cc * neg_mask.unsqueeze(0)  # shape (B,2K,2K)
        # sum along j dimension => (B,2K)
        sum_neg = ex_cc_neg.sum(dim=2) + 1e-6

        # forward_loss_per_child: (B,2K)
        forward_loss_per_child = -torch.log(ex_plus / (ex_plus + sum_neg))
        # final forward_loss: average over batch and child dimension
        forward_loss = forward_loss_per_child.mean()

        # ---------------------------------------------------------------
        # 2) Parent->Child (Reverse)
        # ---------------------------------------------------------------
        # For each parent p_k, the "positives" are c(2k) and c(2k+1).
        # We'll define a parent->child similarity matrix sim_pc: (B,K,2K)
        # sim_pc[b,k,i] = sim( p_mean[b,k], c_mean[b,i] )
        # We'll do a broadcast approach:
        p_mean1 = p_mean.unsqueeze(2)  # (B,K,1,C)
        p_std1  = p_std.unsqueeze(2)   # (B,K,1,D)
        c_mean2 = c_mean.unsqueeze(1)  # (B,1,2K,C)
        c_std2  = c_std.unsqueeze(1)   # (B,1,2K,D)

        # sim_pc: (B, K, 2K)
        sim_pc = self.kl_divergence(p_mean1, p_std1, c_mean2, c_std2)

        # Construct positives for parent k => child(2k) and child(2k+1)
        # Construct negatives => child i in other parents
        # We'll do a (K,2K) mask at once: for parent k, positive indices = {2k, 2k+1}, negative otherwise
        # shape: (K,2K)
        i_idx = torch.arange(2*K, device=c_mean.device)
        parent_grid = torch.arange(K, device=c_mean.device).unsqueeze(1)  # (K,1)
        # child_parent = i_idx // 2 (from above)
        # But we want to define: same_parent_rev(k,i) => True if i//2 == k
        same_parent_rev = (i_idx.unsqueeze(0)//2 == parent_grid)  # shape (K,2K)
        # positives are exactly c(2k) and c(2k+1)
        # negatives are others
        # We can define a single mask: mask_rev(k,i) = True if i//2 != k
        neg_mask_rev = ~same_parent_rev

        # We'll handle "multiple positives" by summing their exponentiated similarities
        # ex_pos = sum_{ i in positives } exp(sim_pc[k,i]/temp)
        # ex_neg = sum_{ i in negatives } exp(sim_pc[k,i]/temp)

        ex_pc = torch.exp(sim_pc / temperature).mean(dim=-1)  # shape (B,K,2K)

        # sum positives: ex_pos[k] = sum_{i in same_parent_rev(k,i)} ex_pc[b,k,i]
        # shape => (B,K)
        ex_pos = (ex_pc * same_parent_rev.unsqueeze(0)).sum(dim=2) + 1e-6

        # sum negatives: ex_neg[k] = sum_{i in neg_mask_rev(k,i)} ...
        ex_neg = (ex_pc * neg_mask_rev.unsqueeze(0)).sum(dim=2) + 1e-6

        # reverse_loss_per_parent: (B,K)
        reverse_loss_per_parent = -torch.log(ex_pos / (ex_pos + ex_neg))
        reverse_loss = reverse_loss_per_parent.mean()

        # ---------------------------------------------------------------
        # Combine Forward + Reverse
        # ---------------------------------------------------------------
        total_loss = forward_loss + reverse_loss
        return total_loss    
    def flops(self, N):
        # calculate flops for 1 group with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops


class StructFormerBlock(nn.Module):
    r""" StructFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        interval (int): Interval for LDA.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        num_patch_size
        impl_type (str): 
        use_extra_conv (bool): Extra convolution layer. Default: True
    """

    def __init__(self,ack, dim, input_resolution, num_heads, group_size=7, interval=8, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1, 
                 pad_type=0, use_extra_conv=True, use_cpe=False, no_mask=False, adaptive_interval=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.interval = interval
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        self.pad_type = pad_type
        self.use_extra_conv = use_extra_conv
        self.use_cpe = use_cpe
        if min(self.input_resolution) <= self.group_size:
            # if group size is larger than input resolution, we don't partition groups
            self.lsda_flag = 0
            self.group_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=(not use_cpe))
        self.clk = ack
        if self.use_cpe:
            self.cpe = nn.Conv2d(in_channels=input_resolution[0], out_channels=input_resolution[0], kernel_size=3, padding=1, groups=input_resolution[0])
            self.norm_cpe = norm_layer(dim)

        if adaptive_interval:
            self.interval = int(np.ceil(self.input_resolution[0] / self.group_size))

        if self.use_extra_conv:
            self.ex_kernel = [3, 3]
            padding = (self.ex_kernel[0] - 1) // 2
            self.ex_conv = nn.Conv2d(dim, dim, self.ex_kernel, padding=padding, groups=dim)
            self.ex_ln = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=True)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # compute attention mask
        attn_mask = None

        if not no_mask:
            H, W = self.input_resolution

            size_div = self.interval * self.group_size if self.lsda_flag == 1 else self.group_size

            pad_w = (size_div - W % size_div) % size_div
            pad_h = (size_div - H % size_div) % size_div

            if self.pad_type == 0:
                pad_l = pad_t = 0
            else:
                pad_l = pad_w // 2
                pad_t = pad_h // 2
            
            pad_r = pad_w - pad_l
            pad_b = pad_h - pad_t

            Hp = H + pad_h
            Wp = W + pad_w

            mask = torch.zeros((1, Hp, Wp, 1))
            if pad_h > 0:
                mask[:, -pad_b:, :, :] = -1
                mask[:, : pad_t, :, :] = -1
            if pad_w > 0:
                mask[:, :, -pad_r:, :] = -1
                mask[:, :, : pad_l, :] = -1

            if self.lsda_flag == 0: # 0 for SDA
                G = Gh = Gw = self.group_size
                nG = Hp * Wp // G**2
                # attn_mask
                if pad_w > 0 or pad_h > 0:
                    mask = mask.reshape(1, Hp // G, G, Wp // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                    mask = mask.reshape(nG, 1, G * G)
                    attn_mask = torch.zeros((nG, G * G, G * G))
                    attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
                else:
                    attn_mask = None
            else: # 1 for LDA
                I = self.interval
                G = Gh = Gw = self.group_size
                Rh, Rw = Hp // (Gh * I), Wp // (Gw * I)
                nG = I ** 2 * Rh * Rw
                # attn_mask
                if pad_w > 0 or pad_h > 0:
                    mask = mask.reshape(1, Rh, Gh, I, Rw, Gw, I, 1).permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
                    mask = mask.reshape(nG, 1, Gh * Gw)
                    attn_mask = torch.zeros((nG, Gh * Gw, Gh * Gw))
                    attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
                else:
                    attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x, token):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        shortcut = x
        shortcut_token = token
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.use_cpe:
            x = x + self.norm_cpe(self.cpe(x))

        # padding
        size_div = self.interval * self.group_size if self.lsda_flag == 1 else self.group_size

        pad_w = (size_div - W % size_div) % size_div
        pad_h = (size_div - H % size_div) % size_div

        if self.pad_type == 0:
            pad_l = pad_t = 0
        else:
            pad_l = pad_w // 2
            pad_t = pad_h // 2
        
        pad_r = pad_w - pad_l
        pad_b = pad_h - pad_t

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # group embeddings
        if self.lsda_flag == 0: # 0 for SDA
            G = Gh = Gw = self.group_size
            x = x.reshape(B, Hp // G, G, Wp // G, G, C).permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(B * Hp * Wp // G**2, G**2, C)
        else: # 1 for LDA
            I = self.interval
            G = Gh = Gw = self.group_size
            Rh, Rw = Hp // (Gh * I), Wp // (Gw * I)
            x = x.reshape(B, Rh, Gh, I, Rw, Gw, I, C).permute(0, 1, 4, 3, 6, 2, 5, 7).contiguous()
            x = x.reshape(B * Rh * Rw * I * I, Gh * Gw, C)

        # multi-head self-attention
        x, token, kl_loss = self.attn(x,token, self.clk, mask=self.attn_mask)  # nW*B, G*G, C

        # ungroup embeddings
        if self.lsda_flag == 0:
            x = x.reshape(B, Hp // G, Wp // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C
        else:
            x = x.reshape(B, Rh, Rw, I, I, Gh, Gw, C).permute(0, 1, 5, 3, 2, 6, 4, 7).contiguous() # B, Rh, Gh, I, Rw, Gw, I, C
        x = x.view(B, Hp, Wp, C)

        # remove padding
        if pad_w > 0 or pad_h > 0:
            x = x[:, pad_t:H+pad_t, pad_l:W+pad_l, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        token = shortcut_token*0.001 + token
        
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        token = token + self.mlp(self.norm2(token))

        if self.use_extra_conv:
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            x = self.ex_conv(x)
            x = x.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()
            x = self.ex_ln(x)

        return x, token, kl_loss

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}, " \
               f"interval={self.interval}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # LSDA
        nW = H * W / self.group_size / self.group_size
        flops += nW * self.attn.flops(self.group_size * self.group_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, patch_size=[2], num_input_patch_size=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(nn.Conv2d(dim, out_dim, kernel_size=ps, 
                                                stride=stride, padding=padding))

    def forward(self, x, token):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = self.norm(x)
        token = self.norm(token)
        tokens = torch.cat([token, token, token, token], dim =-1).view(token.size(0),token.size(2), 2*token.size(1), 2)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        # x = x.view(B, int(L**0.5), , C).permute(0, 3, 1, 2)

        xs = []
        ts = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x).flatten(2).transpose(1, 2)
            # tmp_x = self.reductions[i](x)
            # tmp_x = F.interpolate(tmp_x, scale_factor=2, mode='nearest')
            # tmp_x = tmp_x.flatten(2).transpose(1, 2)
            tmp_token = self.reductions[i](tokens).flatten(2).transpose(1, 2)
            xs.append(tmp_x)
            ts.append(tmp_token)
        x = torch.cat(xs, dim=2)
        token = torch.cat(ts,dim=2)
        return x, token

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                out_dim = 2 * self.dim // 2 ** i
            else:
                out_dim = 2 * self.dim // 2 ** (i + 1)
            flops += (H // 2) * (W // 2) * ps * ps * out_dim * self.dim
        return flops


class Stage(nn.Module):
    """ StructFormer blocks for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        group_size (int): variable G in the paper, one group has GxG embeddings
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,ack, dim, input_resolution, depth, num_heads, group_size, interval,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 patch_size_end=[4], num_patch_size=None, use_cpe=False, pad_type=0, 
                 no_mask=False, adaptive_interval=False, use_acl=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1

            # use extra convolution block every 3 blocks
            use_extra_conv = ((i + 1) % 3 == 0) and (i < depth - 1) and use_acl

            self.blocks.append(StructFormerBlock(ack=ack, dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, group_size=group_size[i], interval=interval,
                                 lsda_flag=lsda_flag,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 num_patch_size=num_patch_size,
                                 use_extra_conv=use_extra_conv,
                                 use_cpe=use_cpe,
                                 pad_type=pad_type,
                                 no_mask=no_mask,
                                 adaptive_interval=adaptive_interval
                                 ))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, 
                                         patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def forward(self, x_total):
        x, token, total_kl_loss = x_total
        for blk in self.blocks:
            if self.use_checkpoint:
                x, token, kl_loss = checkpoint.checkpoint(blk, x, token)
            else:
                x, token, kl_loss = blk(x, token)
            total_kl_loss = total_kl_loss+kl_loss
        if self.downsample is not None:
            x, token = self.downsample(x, token)
        return x, token, total_kl_loss

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: [4].
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[0] // patch_size[0]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # B, K, C = token.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        xs = []
        # token = torch.cat([token,token, token, token],1).view(B,2*K, 2,C)
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            # tox = self.projs[0](token)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = 0
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                dim = self.embed_dim // 2 ** i
            else:
                dim = self.embed_dim // 2 ** (i + 1)
            flops += Ho * Wo * dim * self.in_chans * (self.patch_size[i] * self.patch_size[i])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class StructFormer(nn.Module):
    r""" StructFormer
        A PyTorch impl of : `StructFormer: A Versatile Vision Transformer Based on Structured Attention`  -

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        group_size (int): Group size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        use_cpe (bool): Whether to use conditional positional encoding. Default: False
        group_type (str): Strategy to change the group size in different stages. Default: constant
        pad_type (bool): 0 to pad in one direction, otherwise 1. Default: 0
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 group_size=[7, 7, 7, 7], crs_interval=[8, 4, 2, 1], mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, merge_size=[[2], [2], [2]], use_cpe=False,
                 group_type='constant', pad_type=0, no_mask=False,
                 adaptive_interval=False, use_acl=False,  **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_token = 3
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.token_mu = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.token_sigma = nn.Parameter(torch.rand(1, 1, embed_dim))
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # compute group size for each layer
        group_size = self.compute_group_size(group_size, depths, patches_resolution, group_type)

        # build layers
        self.layers = nn.ModuleList()

        num_patch_sizes = [len(patch_size)] + [len(m) for m in merge_size]
        for i_layer in range(self.num_layers):
            # if i_layer == 3:
            #     ack=3
            # elif i_layer == 2:
            #     ack=2
            # elif i_layer == 1:
            #     ack=1
            # else:
            #     ack=0
            patch_size_end = merge_size[i_layer] if i_layer < self.num_layers - 1 else None
            num_patch_size = num_patch_sizes[i_layer]
            layer = Stage(ack=i_layer, dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               group_size=group_size[i_layer],
                               interval=crs_interval[i_layer],
                               mlp_ratio=self.mlp_ratio[i_layer],
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               patch_size_end=patch_size_end,
                               num_patch_size=num_patch_size,
                               use_cpe=use_cpe,
                               pad_type=pad_type,
                               no_mask=no_mask,
                               adaptive_interval=adaptive_interval,
                               use_acl=use_acl)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def compute_group_size(self, group_size=[7, 7, 7, 7], depths=[2, 2, 6, 2], resolution=[56, 56], group_type='constant'):
        r"""genenrate group size for structformer
        
        output:
            - rst_group_size: should be in the shape [[4], [4, 4], [14, 14, 14], [7, 7]] if the depths = [1, 2, 3, 2]
        """
        rst_group_size = []

        # compute linear fraction patch size
        min_size = 4
        total_depth = sum(depths)
        step_size = (1 - min_size / resolution[0]) / total_depth
        group_fraction = np.arange(min_size / resolution[0], 1.0, step_size)
    
        cnt = 0
        for i_stage in range(len(depths)):
            rst_group_size.append([])
            cur_resolution = resolution[0] // 2 ** i_stage
            for i_block in range(depths[i_stage]):
                if group_type == 'constant':
                    # constant group size for each stage
                    rst_group_size[i_stage].append(group_size[i_stage])
                elif group_type == 'linear':
                    # the fraction of group size relative to input resolution grow in linear
                    gz = cur_resolution * group_fraction[cnt]
                    rst_group_size[i_stage].append(max(4, int(np.ceil(gz))))
                elif group_type == 'linear_div':
                    # if fraction > 1/2, let fraction = 1/2 if fraction < 3/4 else 1
                    gz = cur_resolution * group_fraction[cnt]
                    if gz > cur_resolution // 2:
                        gz = cur_resolution if gz > cur_resolution * 3 / 4 or i_stage != 2 else cur_resolution // 2
                    rst_group_size[i_stage].append(max(4, int(np.ceil(gz))))
                elif group_type == 'alter':
                    # if fraction > 1/2, let fraction alter between 1/2 and 1
                    gz = cur_resolution * group_fraction[cnt]
                    if gz > cur_resolution // 2:
                        gz = cur_resolution if cnt % 2 != 0 or i_stage != 2 else cur_resolution // 2
                    rst_group_size[i_stage].append(max(4, int(np.ceil(gz))))
                elif group_type == '7_14':
                    rst_group_size[i_stage].append(group_size[i_stage] if i_stage != 2 or i_block >= 4 else group_size[i_stage] // 2) 
                cnt += 1

        print("Group Size:")
        print(rst_group_size)

        return rst_group_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, return_layer_outputs=False):
        kl_loss = 0
        layer_outputs = [] if return_layer_outputs else None
        
        mu = self.token_mu.expand(x.size(0), self.num_token, -1)
        sigma = self.token_sigma.expand(x.size(0), self.num_token, -1)
        tokens = torch.normal(mu, sigma)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_total = [x, tokens, kl_loss]
        
        for i, layer in enumerate(self.layers):
            x_total = layer(x_total)
            if return_layer_outputs:
                x_temp, tokens_temp, _ = x_total
                layer_outputs.append({
                    'layer_idx': i,
                    'patch_embeddings': x_temp.clone(),
                    'semantic_tokens': tokens_temp.clone()
                })
        
        x, tokens, kl_loss = x_total
        x = self.norm(x)  # B L C
        tokens = self.norm(tokens)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        
        if return_layer_outputs:
            return x, tokens, kl_loss, layer_outputs
        return x,tokens, kl_loss

    def forward(self, x, return_layer_outputs=False):
        if return_layer_outputs:
            x, token, kl_loss, layer_outputs = self.forward_features(x, return_layer_outputs=True)
        else:
            x,token, kl_loss = self.forward_features(x)
        cos_sim = F.cosine_similarity(x.unsqueeze(1), token, dim=2)*20  # shape: [B, K]

        # softmax로 정규화하여 weight 생성 (token dimension에 대해 softmax 적용)
        attn_weight = F.softmax(cos_sim, dim=1).unsqueeze(2)  # shape: [B, K, 1]

        # token에 softmax weight를 곱하여 업데이트
        token = attn_weight * token
        sim_token = self.avgpool(token.transpose(1, 2))
        sim_token = torch.flatten(sim_token, 1)
        x = self.head(x)
        sim_token = self.head(sim_token)
        
        if return_layer_outputs:
            return x, sim_token, kl_loss, layer_outputs
        return x,sim_token, kl_loss

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
