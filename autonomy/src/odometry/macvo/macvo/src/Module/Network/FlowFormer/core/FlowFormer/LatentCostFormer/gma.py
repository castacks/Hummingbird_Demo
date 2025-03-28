import torch
from torch import nn, einsum
# from einops import rearrange


class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        h_xu, h_d = height_emb.shape
        h_x, h_u = h, h_xu // h
        height_emb = height_emb.view(h_x, h_u, h_d).unsqueeze(-2)
        # height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        
        w_yv, w_d = width_emb.shape
        w_y, w_v = w, w_yv // w
        width_emb = width_emb.view(w_y, w_v, w_d).unsqueeze(-3)
        # width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(
        self,
        *,
        args,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads = self.heads

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        # q = rearrange(q, 'b (h d) x y -> b h x y d', h=heads)
        q_b, q_hd, q_x, q_y = q.size(0), q.size(1), q.size(2), q.size(3)
        q_h, q_d = heads, q_hd // heads
        q = q.view(q_b, q_h, q_d, q_x, q_y).permute(0, 1, 3, 4, 2)
        
        # k = rearrange(k, 'b (h d) x y -> b h x y d', h=heads)
        k_b, k_hd, k_x, k_y = k.size(0), k.size(1), k.size(2), k.size(3)
        k_h, k_d = heads, k_hd // heads
        k = k.view(k_b, k_h, k_d, k_x, k_y).permute(0, 1, 3, 4, 2)
        
        q = self.scale * q

        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        # sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        sim_b, sim_h, sim_x, sim_y, sim_u, sim_v = sim.shape
        sim = sim.view(sim_b, sim_h, sim_x * sim_y, sim_u * sim_v)
        
        attn = sim.softmax(dim=-1)

        return attn


class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads = self.heads
        h, w = fmap.size(2), fmap.size(3)

        v = self.to_v(fmap)
        # v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        v_b, v_hd, v_x, v_y = v.size(0), v.size(1), v.size(2), v.size(3)
        v_h, v_d = heads, v_hd // heads
        v = v.view(v_b, v_h, v_d, v_x, v_y).permute(0, 1, 3, 4, 2).view(v_b, v_h, v_x * v_y, v_d)

        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = torch.matmul(attn, v)

        # out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out_b, out_h, out_d = out.size(0), out.size(1), out.size(3)
        out_x, out_y = h, w
        out = out.view(out_b, out_h, out_x, out_y, out_d).permute(0, 1, 4, 2, 3).view(out_b, out_h * out_d, out_x, out_y)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out
