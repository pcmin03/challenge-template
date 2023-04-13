import torch
from torch import nn
from utils.common import mean_std, LIPS_UNITS, HANDS_UNITS, POSE_UNITS, UNITS, INPUT_SIZE, LIPS_START, N_COLS

LIPS_MEAN = mean_std['lips_mean']
LIPS_STD = mean_std['lips_std']
LEFT_HANDS_MEAN = mean_std['left_hands_mean']
LEFT_HANDS_STD = mean_std['left_hands_std']
RIGHT_HANDS_MEAN = mean_std['right_hands_mean']
RIGHT_HANDS_STD = mean_std['right_hands_std']
POSE_MEAN = mean_std['pose_mean']
POSE_STD = mean_std['pose_std']
# Epsilon value for layer normalisation
LAYER_NORM_EPS = 1e-6

# Dense layer units for landmarks
LIPS_UNITS = 384
HANDS_UNITS = 384
POSE_UNITS = 384
# final embedding and transformer embedding size

# Transformer
NUM_OF_HEADS= 8
NUM_BLOCKS = 2
MLP_RATIO = 2

# Dropout
EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
CLASSIFIER_DROPOUT_RATIO = 0.10


class LandmarkEmbedding(nn.Module):
    def __init__(self, units, name, input_dim, INPUT_SIZE):
        super(LandmarkEmbedding, self).__init__()
        self.units = units
        self.name = name
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = nn.Parameter(torch.zeros(INPUT_SIZE, self.units), requires_grad=False)
        # Embedding
        self.dense = nn.Sequential(
            nn.Linear(input_dim, self.units, bias=False),
            nn.ReLU(),
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
    
class Embedding(nn.Module):
    def __init__(self,units):
        super(Embedding, self).__init__()

        self.positional_embedding = nn.Embedding(INPUT_SIZE+1, units)
        nn.init.zeros_(self.positional_embedding.weight)
        
        self.lips_embedding = LandmarkEmbedding(LIPS_UNITS, 'lips', 80, INPUT_SIZE)
        self.left_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'left_hand', 42, INPUT_SIZE)
        self.right_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'right_hand', 42, INPUT_SIZE)
        self.pose_embedding = LandmarkEmbedding(POSE_UNITS, 'pose', 20, INPUT_SIZE)
        
        self.landmark_weights = nn.Parameter(torch.zeros([4], dtype=torch.float32), requires_grad=True)
        
        self.fc = nn.Sequential(
            nn.Linear(LIPS_UNITS, units, bias=False),
            nn.ReLU(),
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
        diffs = torch.reshape(diffs, [-1, INPUT_SIZE, S*S])
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
            torch.tensor(INPUT_SIZE, dtype=torch.int32).cuda(),
            (torch.floor_divide(
                non_empty_frame_idxs, 
                torch.max(non_empty_frame_idxs, dim=1, keepdim=True).values) * INPUT_SIZE).type(torch.int32).cuda(),
        ).to(torch.int64)
        
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
       
        return x
    

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

def torch_input(frames,non_empty_frame_idxs):
    
    TLIPS_MEAN = torch.tensor(LIPS_MEAN).cuda()
    TLIPS_STD = torch.tensor(LIPS_STD).cuda()
    TLEFT_HANDS_MEAN = torch.tensor(LEFT_HANDS_MEAN).cuda()
    TLEFT_HANDS_STD = torch.tensor(LEFT_HANDS_STD).cuda()
    TRIGHT_HANDS_MEAN = torch.tensor(RIGHT_HANDS_MEAN).cuda()
    TRIGHT_HANDS_STD = torch.tensor(RIGHT_HANDS_STD).cuda()
    TPOSE_MEAN = torch.tensor(POSE_MEAN).cuda()
    TPOSE_STD = torch.tensor(POSE_STD).cuda()
    print(frames.dtype,)
    x = frames
    x = x[:, :INPUT_SIZE, :N_COLS, :2]
    # LIPS
    lips = x[:,:INPUT_SIZE,LIPS_START:LIPS_START+40,:2]
    
    lips = torch.where(
            lips == 0.0,
            torch.tensor(0.),
            (lips - TLIPS_MEAN) / TLIPS_STD,
        )
    a = (lips - TLIPS_MEAN) / TLIPS_STD
    lips = torch.reshape(lips, [-1, INPUT_SIZE, 40*2])
    # LEFT HAND
    left_hand = x[:,:INPUT_SIZE,40:40+21,:2]
    left_hand = torch.where(
            left_hand == 0.0,
            torch.tensor(0.),
            (left_hand - TLEFT_HANDS_MEAN) / TLEFT_HANDS_STD,
        )
    left_hand = torch.reshape(left_hand, [-1, INPUT_SIZE, 21*2])
    # RIGHT HAND
    right_hand = x[:,:INPUT_SIZE,61:61+21,:2]
    right_hand = torch.where(
            right_hand == 0.0,
            torch.tensor(0.).cuda(),
            (right_hand - TRIGHT_HANDS_MEAN) / TRIGHT_HANDS_STD,
        )
    right_hand = torch.reshape(right_hand, [-1, INPUT_SIZE, 21*2])
    # POSE
    pose = x[:,:INPUT_SIZE,82:82+10,:2]
    pose = torch.where(
            pose == 0.0,
            torch.tensor(0.).cuda(),
            (pose - TPOSE_MEAN) / TPOSE_STD,
        )
    pose = torch.reshape(pose, [-1, INPUT_SIZE, 10*2])

    lips = lips.cuda()
    left_hand = left_hand.cuda()
    right_hand = right_hand.cuda()
    pose = pose.cuda()
    non_empty_frame_idxs = non_empty_frame_idxs.cuda()

    x = lips, left_hand, right_hand, pose, non_empty_frame_idxs

    mask = torch.ne(non_empty_frame_idxs, -1).float()
    mask = mask[:,:,None]
#     mask = torch.unsqueeze(mask, dim=2)
    return x, mask

class Transformer(torch.nn.Module):
    def __init__(self, num_blocks, units, num_classes):
        super(Transformer, self).__init__()
        
        self.Embedding = Embedding(units)
        
        self.num_blocks = num_blocks
        # First Layer Normalisation
        self.ln_1s = torch.nn.ModuleList([torch.nn.LayerNorm(units, eps=LAYER_NORM_EPS) for i in range(num_blocks)])
        # Multi Head Attention
        self.mhas = torch.nn.ModuleList([MultiHeadAttention(units, NUM_OF_HEADS) for i in range(num_blocks)])
        # Second Layer Normalisation
        self.ln_2s = torch.nn.ModuleList([torch.nn.LayerNorm(units, eps=LAYER_NORM_EPS) for i in range(num_blocks)])
        # Multi Layer Perception
        self.mlps = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(units, units * MLP_RATIO),
            torch.nn.ReLU(),
            torch.nn.Dropout(MLP_DROPOUT_RATIO),
            torch.nn.Linear(units * MLP_RATIO, units),
        ) for i in range(num_blocks)])
        
        nn.init.xavier_uniform_(self.mlps[0][0].weight)
        nn.init.kaiming_uniform_(self.mlps[0][3].weight)
        
        self.dropout = nn.Dropout(p=CLASSIFIER_DROPOUT_RATIO)
        
        self.fc = nn.Linear(units, num_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self,frames, non_empty_idx):
        x, attention_mask = torch_input(frames, non_empty_idx)
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