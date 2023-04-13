import torch
from torch import nn
import torch.nn.functional as F

# Dropout
EMBEDDING_DROPOUT = 0.00
CLASSIFIER_DROPOUT_RATIO = 0.10

def scaled_dot_product(q,k,v, attention_mask):
    #calculates Q . K(transpose)
    qkt = torch.matmul(q,k.transpose(-1,-2))
    #caculates scaling factor
    dk = torch.sqrt(torch.tensor(q.shape[-1],dtype=torch.float32))
    scaled_qkt = qkt/dk
    softmax = F.softmax(scaled_qkt.masked_fill(attention_mask == 0, -1e9), dim=-1)
    
    z = torch.matmul(softmax,v)
    #shape: (m,Tx,depth), same shape as q,k,v
    return z

class LandmarkEmbedding(nn.Module):
    def __init__(self, units, name, input_dim, input_size):
        super(LandmarkEmbedding, self).__init__()
        self.units = units
        self.name = name
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = nn.Parameter(torch.zeros(input_size, self.units), requires_grad=False)
        # Embedding
        self.dense = nn.Sequential(
            nn.Linear(input_dim, self.units, bias=False),
            nn.GELU(),
            nn.Linear(self.units, self.units, bias=False)
        )

        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.kaiming_uniform_(self.dense[2].weight)
    
    def forward(self, x):
        return torch.where(
            # Checks whether landmark is missing in frame
            torch.sum(x, dim=2, keepdim=True) == 0,
            # If so, the empty embedding is used
            self.empty_embedding,
            # Otherwise the landmark data is embedded
            self.dense(x),
        ).cuda()

class Embedding(nn.Module):
    def __init__(self, units, input_size):
        super(Embedding, self).__init__()
        self.input_size = input_size
        self.positional_embedding = nn.Embedding(input_size+1, units)
        nn.init.zeros_(self.positional_embedding.weight)
        LIPS_units = 384
        HANDS_units = 384
        POSE_units = 384

        self.lips_embedding = LandmarkEmbedding(LIPS_units, 'lips', 80, self.input_size)
        self.left_hand_embedding = LandmarkEmbedding(HANDS_units, 'left_hand', 42, self.input_size)
        self.right_hand_embedding = LandmarkEmbedding(HANDS_units, 'right_hand', 42, self.input_size)
        self.pose_embedding = LandmarkEmbedding(POSE_units, 'pose', 20, self.input_size)
        
        self.landmark_weights = nn.Parameter(torch.zeros([4], dtype=torch.float32), requires_grad=True)
        
        self.fc = nn.Sequential(
            nn.Linear(units, units, bias=False),
            nn.GELU(),
            nn.Linear(units, units, bias=False)
        )

        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.kaiming_uniform_(self.fc[2].weight)

    def get_diffs(self, l):
        S = l.shape[2]
        other = torch.unsqueeze(l, 3)
        other = other.repeat(1, 1, 1, S)
        other = torch.transpose(other, 2, 3)
        diffs = torch.unsqueeze(l, 3) - other
        diffs = torch.reshape(diffs, [-1, self.input_size, S*S])
        return diffs

    def forward(self, lips0, left_hand0, right_hand0, pose0, non_empty_frame_idxs):
        lips_embedding = self.lips_embedding(lips0)
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        right_hand_embedding = self.right_hand_embedding(right_hand0)
        pose_embedding = self.pose_embedding(pose0)

        
        x = torch.stack((lips_embedding, left_hand_embedding, right_hand_embedding, pose_embedding), dim=3)
        x = x * F.softmax(self.landmark_weights, dim=0)
        x = torch.sum(x, dim=3)
  
        
        x = self.fc(x)
        
        normalised_non_empty_frame_idxs = torch.where(
            torch.eq(non_empty_frame_idxs, -1.0),
            torch.tensor(self.input_size, dtype=torch.int32).cuda(),
            (torch.floor_divide(
                non_empty_frame_idxs, torch.max(non_empty_frame_idxs, dim=1, keepdim=True).values) * self.input_size).cuda(),
        ).to(torch.int64)
        
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
       
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,d_model,num_of_heads):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model//num_of_heads
        self.wq = torch.nn.ModuleList([torch.nn.Linear(d_model, self.depth) for i in range(num_of_heads)])
        self.wk = torch.nn.ModuleList([torch.nn.Linear(d_model, self.depth) for i in range(num_of_heads)])
        self.wv = torch.nn.ModuleList([torch.nn.Linear(d_model, self.depth) for i in range(num_of_heads)])
        self.wo = torch.nn.Linear(num_of_heads*self.depth,d_model)
        
    def forward(self,x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(scaled_dot_product(Q,K,V, attention_mask))
        
        multi_head = torch.cat(multi_attn, dim=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention

class Transformer(torch.nn.Module):
    def __init__(self, num_blocks, mlp_ratio, mlp_dropout_ratio, units, num_classes, input_size):
        super(Transformer, self).__init__()
        
        self.Embedding = Embedding(units, input_size).cuda()
        
        self.num_blocks = num_blocks
        # First Layer Normalisation
        self.ln_1s = torch.nn.ModuleList([torch.nn.LayerNorm(units) for i in range(num_blocks)])
        # Multi Head Attention
        self.mhas = torch.nn.ModuleList([MultiHeadAttention(units, 8) for i in range(num_blocks)])
        # Second Layer Normalisation
        self.ln_2s = torch.nn.ModuleList([torch.nn.LayerNorm(units) for i in range(num_blocks)])
        # Multi Layer Perception
        self.mlps = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(units, units * mlp_ratio),
            torch.nn.GELU(),
            torch.nn.Dropout(mlp_dropout_ratio),
            torch.nn.Linear(units * mlp_ratio, units),
        ) for i in range(num_blocks)])
        
        nn.init.xavier_uniform_(self.mlps[0][0].weight)
        nn.init.kaiming_uniform_(self.mlps[0][3].weight)
        
        self.fc = nn.Linear(units, num_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x, attention_mask):
        x = self.Embedding(x[0], x[1], x[2], x[3], x[4])
        # Iterate input over transformer blocks
        for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s, self.mlps):
            x1 = ln_1(x)
            attention_output = mha(x1, attention_mask)
            x2 = x1 + attention_output
            x3 = ln_2(x2)
            x3 = mlp(x3)
            x = x3 + x2
            
        # Pooling
        x = torch.sum(x * attention_mask, axis=1) / torch.sum(attention_mask, axis=1)
        # Classification Layer
        x = self.fc(x)
        return x