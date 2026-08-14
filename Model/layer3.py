import torch
import torch.nn as nn
import torch.nn.functional as F

class EventDetector(nn.Module):
    def __init__(self, e_dim=128, events=20):
        super().__init__()
        self.W1 = nn.Linear(e_dim, events, bias=False)

    def forward(self,e):    #e는 [B,L,128] 형태로 들어오고 beta는 [B,L,20]
        beta = F.softmax(self.W1(e), dim=-1)
        e_split = beta.unsqueeze(-1) * e.unsqueeze(-2)
        return beta,e_split




class EventTransitionNet(nn.Module):
    def __init__(self, e_dim=128, events =20, ctx_len=4, pos_dim=32, attn_dim=64, user_dim=128):
        # F (4,160) 형태
        super().__init__()
        self.pos_emb = nn.Embedding(ctx_len, pos_dim)
        self.W2 = nn.Linear(e_dim +pos_dim, attn_dim)
        self.W3  =nn.Linear(attn_dim, 1, bias=False)
        self.W4 = nn.Linear(e_dim+user_dim,e_dim)

    def build_R(self, e, e_split, pad_mask=None):
        
        batch, ctx_len, _ = e.shape
        p = self.pos_emb(torch.arange(ctx_len, device=e.device))        # [L,32]
        f = torch.cat([e, p.unsqueeze(0).expand(batch, -1, -1)], -1)  # f_i = [e_i ; p_i]

        gamma = self.W3(torch.tanh(self.W2(f))).squeeze(-1)       # Eq 15공식
        
        if pad_mask is not None:
            gamma = gamma.masked_fill(~pad_mask, -1e9)            # 패딩 자리 제외
        gamma = F.softmax(gamma, dim=-1)                          # [B,L]
        R = torch.einsum("bl,blkd->bkd", gamma, e_split)          # Eq 16
        return R, gamma

    def activate(self, R, e_t, u):

        delta = F.softmax(torch.einsum("bkd,bcd->bck", R, e_t), -1)   # Eq 17
        c = torch.einsum("bck,bkd->bcd", delta, R)                    # Eq 18
        u = u.unsqueeze(1).expand(-1, c.size(1), -1)
        c_u = torch.tanh(self.W4(torch.cat([c, u], -1)))              # Eq 19
        return c_u, delta


class NextNewsPredictor(nn.Module):
    def forward(self,c_u, e_t):
        return (c_u*e_t).sum(-1)

    @staticmethod
    def recommend(logits, fake_prob, topk=5, threshold=0.5):
        score = torch.sigmoid(logits)
        score = score.masked_fill(fake_prob>=threshold, -1)
        return score.topk(min(topk, score.size(-1)),dim=-1)
