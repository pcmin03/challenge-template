import torch
import torch.nn as nn
import torch.nn.functional as F

### LandmarkEmbedding
class LandmarkEmbedding(nn.Module):
    def __init__(self, units, input_each_dim, INPUT_SIZE):
        super(LandmarkEmbedding, self).__init__()
        
        self.units = units
        self.input_each_dim = input_each_dim
        self.input_size = INPUT_SIZE
        
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = nn.Parameter(torch.zeros(self.input_size , self.units), requires_grad=False)
        # Embedding
        self.dense = nn.Sequential(
            nn.Linear(self.input_each_dim, self.units, bias=False),
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
        )
        
### Embedding
class Embedding(nn.Module):
    def __init__(self, INPUT_SIZE, UNITS, LIPS_UNITS, HANDS_UNITS, POSE_UNITS,
                 LIPS_DIM: int = 80, HANDS_DIM: int = 42, POSE_DIM: int = 20):
        super(Embedding, self).__init__()
        
        self.input_size = INPUT_SIZE
        self.units = UNITS
        self.lips_units = LIPS_UNITS
        self.hands_units = HANDS_UNITS
        self.pose_units = POSE_UNITS
        self.lips_dim = LIPS_DIM
        self.hands_dim = HANDS_DIM
        self.pose_dim = POSE_DIM

        self.positional_embedding = nn.Embedding(self.input_size+1, self.units)
        nn.init.zeros_(self.positional_embedding.weight)
        
        self.lips_embedding = LandmarkEmbedding(self.lips_units , self.lips_dim, self.input_size)
        self.left_hand_embedding = LandmarkEmbedding(self.hands_units, self.hands_dim, self.input_size)
        self.right_hand_embedding = LandmarkEmbedding(self.hands_units, self.hands_dim, self.input_size)
        self.pose_embedding = LandmarkEmbedding(self.pose_units, self.pose_dim, self.input_size)
        
        self.landmark_weights = nn.Parameter(torch.zeros([4], dtype=torch.float32), requires_grad=True)
        
        self.fc = nn.Sequential(
            nn.Linear(self.lips_units , self.units, bias=False),
            nn.GELU(),
            nn.Linear(self.units, self.units, bias=False)
        )

        nn.init.xavier_uniform_(self.fc[0].weight)
        nn.init.kaiming_uniform_(self.fc[2].weight)


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
    
    
### MultiHeadAttention
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

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, units, num_of_heads):
        super(MultiHeadAttention,self).__init__()
        self.units = units
        self.num_of_heads = num_of_heads
        self.depth = units//num_of_heads
        self.wq = torch.nn.ModuleList([torch.nn.Linear(self.units , self.depth) for i in range(num_of_heads)])
        self.wk = torch.nn.ModuleList([torch.nn.Linear(self.units , self.depth) for i in range(num_of_heads)])
        self.wv = torch.nn.ModuleList([torch.nn.Linear(self.units , self.depth) for i in range(num_of_heads)])
        self.wo = torch.nn.Linear(num_of_heads*self.depth, self.units)
        
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
    
### Transformer
class Transformer(torch.nn.Module):
    def __init__(self, INPUT_SIZE, num_blocks, LIPS_UNITS, HANDS_UNITS,
                 POSE_UNITS, UNITS, NUM_CLASSES, LAYER_NORM_EPS, NUM_OF_HEADS,
                 MLP_RATIO, MLP_DROPOUT_RATIO, CLASSIFIER_DROPOUT_RATIO):
        super(Transformer, self).__init__()
        
        self.input_size = INPUT_SIZE
        self.num_blocks = num_blocks
        self.units = UNITS
        self.lips_units = LIPS_UNITS
        self.hands_units = HANDS_UNITS
        self.pose_units = POSE_UNITS
        self.num_classes = NUM_CLASSES
        self.layer_norm_eps = LAYER_NORM_EPS
        self.num_of_heads = NUM_OF_HEADS
        self.mlp_ratio = MLP_RATIO
        self.mlp_dropout_ratio = MLP_DROPOUT_RATIO
        self.classifier_dropout_ratio = CLASSIFIER_DROPOUT_RATIO
        
        
        self.Embedding = Embedding(self.input_size, self.units, 
                                   self.lips_units, self.hands_units,
                                   self.pose_units)
        
        # First Layer Normalisation
        self.ln_1s = torch.nn.ModuleList([torch.nn.LayerNorm(self.units, eps=self.layer_norm_eps) for i in range(self.num_blocks)])
        # Multi Head Attention
        self.mhas = torch.nn.ModuleList([MultiHeadAttention(self.units, self.num_of_heads) for i in range(self.num_blocks)])
        # Second Layer Normalisation
        self.ln_2s = torch.nn.ModuleList([torch.nn.LayerNorm(self.units, eps=self.layer_norm_eps) for i in range(self.num_blocks)])
        # Multi Layer Perception
        self.mlps = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(self.units, self.units * self.mlp_ratio),
            torch.nn.GELU(),
            torch.nn.Dropout(self.mlp_dropout_ratio),
            torch.nn.Linear(self.units * self.mlp_ratio, self.units),
        ) for i in range(self.num_blocks)])
        
        nn.init.xavier_uniform_(self.mlps[0][0].weight)
        nn.init.kaiming_uniform_(self.mlps[0][3].weight)
        
        self.dropout = nn.Dropout(p=self.classifier_dropout_ratio)
        
        self.fc = nn.Linear(self.units, self.num_classes)

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
        # Dropout
        x = self.dropout(x)
        # Classification Layer
        x = self.fc(x)
        return x