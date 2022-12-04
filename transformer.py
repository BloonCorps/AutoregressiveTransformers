import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from torch import nn, einsum
from torch.utils.data import DataLoader

#################################
### Unconditional Transformer ###
#################################

class UnconditionalDecoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, dropout, 
                 num_heads, qk_depth, v_depth, psuedolikelihood): #num_decoder_experts, expert_hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.attn = Attn(hidden_size, num_heads, qk_depth, v_depth, psuedolikelihood)
        self.dropout = nn.Dropout(p=dropout)
        
        self.layernorm_attn = nn.LayerNorm([self.hidden_size], eps=1e-6, elementwise_affine=True)
        self.layernorm_attn2 = nn.LayerNorm([self.hidden_size], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([self.hidden_size], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=True),
                                 nn.GELU(),
                                 nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=True))
        '''
        self.MoE = MoE(
            dim = self.hidden_size,
            num_experts = decoder_experts, #increase the expert of your model without increasing computation
            hidden_dim = self.hidden_size * 4, #size of hidden dimension in each expert, defaults to 4*dimension
            activation = nn.GELU, # use your preferred activation, will default to GELU
            second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
            second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given #threshold) | random (if gate value > threshold * random_uniform(0, 1))
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25, # experts have fixed capacity per batch. we need some extra capacity in #case gating is not perfectly balanced.
            capacity_factor_eval = 2., # capacity_factor_* should be set to a value >=1
            loss_coef = 1e-2 # multiplier on the auxiliary expert balancing auxiliary loss
        ).to(device)
        '''
    def forward(self, X, encoder_output=None):
        y, weights = self.attn(X)
        X = self.layernorm_attn(self.dropout(y) + X)
        #y, _ = self.MoE(X)
        y = self.ffn(X)
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X, weights

########################
### Attention Module ###
########################

class Attn(nn.Module):
    def __init__(self, hidden_size, num_heads, qk_depth, v_depth, psuedolikelihood):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kd = qk_depth*self.num_heads
        self.vd = v_depth*self.num_heads

        self.psuedolikelihood = psuedolikelihood

        self.q_dense = nn.Linear(self.hidden_size, self.kd, bias=False)
        self.k_dense = nn.Linear(self.hidden_size, self.kd, bias=False)
        self.v_dense = nn.Linear(self.hidden_size, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, self.hidden_size, bias=False)
        
        assert self.kd % self.num_heads == 0
        assert self.vd % self.num_heads == 0

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v, weights 

    def forward(self, X, return_attention=True):
        #if use_encoder_output == False: 
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)

        #split to shape [batch_size, num_heads, len, depth / num_heads]
        q = q.view(q.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (self.num_heads, self.kd // self.num_heads)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (self.num_heads, self.vd // self.num_heads)).permute([0, 2, 1, 3])
        q *= (self.kd // self.num_heads) ** (-0.5) #normalized dot product or something
        
        if self.psuedolikelihood == False: #exact log-likelihood, autoregressive, shifting to 
            #take these facts into account
            bias = -1e10*torch.triu(torch.ones(X.shape[1], X.shape[1]), 1).to(X.device)

        elif self.psuedolikelihood == True: #psuedolikelihood
            bias = -1e10*torch.diag(torch.ones(X.shape[1])).to(X.device)
        
        result, weights = self.dot_product_attention(q, k, v, bias=bias)   
        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        
        if return_attention==True:
            return result, weights #where weights refer to attention weights

        return result

##########################
### Actual Transformer ###
##########################

class UnconditionalTransformer(nn.Module):
    def __init__(self, seq_len=300, hidden_size=512, num_bins=600, dropout=0.00, dnlayers=6, batch_size=512, ffn_hidden_size=2048, 
                 num_heads=4, qk_depth=128, v_depth=128, pseudolikelihood=False, device=torch.device("cuda")):
        super(UnconditionalTransformer, self).__init__()
        #model specific params
        self.hidden_size = hidden_size
        self.num_bins = num_bins
        self.dropout = dropout
        self.dnlayers = dnlayers
        self.num_decoder_dim = seq_len
        self.pseudolikelihood = pseudolikelihood
        self.device = device
        #data specific params
        self.batch_size = batch_size
        #functions
        self.output_function = torch.nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()
        self.loss_function_no_sum = nn.CrossEntropyLoss(reduction="none")
        #model components
        self.embeds = nn.Embedding(1, self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout)
        self.output_dense = nn.Linear(self.hidden_size, self.num_bins, bias=True)
        
        self.decoderlayers = nn.ModuleList([UnconditionalDecoderLayer(hidden_size, ffn_hidden_size, dropout, 
                 num_heads, qk_depth, v_depth, pseudolikelihood) for _ in range(self.dnlayers)])
        
        self.pos_embedder_1 = torch.nn.Embedding(self.num_decoder_dim, 
            self.hidden_size) #decoder pos embedding
    
    def shift_and_pad_(self, X): 
        """For the purpose of autoregressive property. Shift inputs over by 1 and pad.
        x should be [256, 19, 8]. Pad 2nd to last dimension, 
        since 1st dimension is batch and last is embedding."""
        X = X[:, :-1, :]
        X = F.pad(X, (0, 0, 1, 0)) #kinda need to figure out how this works
        return X

    def forward(self, decoder_input=None, sampling=False, parallel_sampling=(False, 0, None), return_attention=False):            
        if parallel_sampling[0] == True:
            return self.sample(batch_size = parallel_sampling[1])
    
        #not used in training, but used by sampling
        if sampling: #sampling the 30 multimodal distributions
            curr_infer_length = decoder_input.shape[1]
            decoder_input = F.pad(decoder_input, (0, self.num_decoder_dim - curr_infer_length))  
            
        #apply embedding and shift and pad, only autoregressive needs shifting
        if self.pseudolikelihood == False:
            decoder_input = self.shift_and_pad_(self.embeds(decoder_input) * (self.hidden_size ** 0.5))
        #if psuedolikelihood, then no shifting needed 
        elif self.pseudolikelihood == True:
            decoder_input = self.embeds(decoder_input) * (self.hidden_size ** 0.5) 
        else:
            raise NotImplementedError #can only be autoregressive or pseudolikelihood
        
        #apply positional embedding
        pos_index = torch.arange(0, self.num_decoder_dim).repeat(decoder_input.size()[0], 1).to(self.device)
        positional_encoding = self.pos_embedder_1(pos_index)
        decoder_input = decoder_input + self.pos_embedder_1(pos_index)
        
        #pass through decoder layers
        weights_list = []
        for layer in self.decoderlayers:
            decoder_input, weights = layer(decoder_input)
            weights_list.append(weights) 
        decoder_output = self.output_dense(decoder_input) 
        
        #decoder_output has dims [batch_size, decoder_dim, num_bins]
        reshaped_decoder_output = decoder_output.view(self.batch_size, -1)
        
        if return_attention == False:
            return decoder_output
        
        #where dof is degrees of freedom
        weights_list = torch.cat(weights_list, dim=1) #shape of [batch_size, num_layers*num_heads, 
            #dof, dof]
        return decoder_output, weights_list

    def sample(self, batch_size, detach=False): #batch size is num samples
        """sampling procedure does not require gradients"""
        total_len = self.num_decoder_dim
        samples = torch.zeros((batch_size, 1)).long().to(self.device)
        
        if detach == True:
            samples = samples.detach()

        for curr_infer_length in range(total_len):
            outputs = self.forward(decoder_input=samples, sampling=True, get_multivar=False)
            outputs = outputs[:, curr_infer_length]
            categorical = self.output_function(outputs) #outputs is energy
            temp_distribution = dist.Categorical(torch.squeeze(categorical))
            x = temp_distribution.sample()
            x = x.unsqueeze(dim=1)#print(categorical.size()) = [10, 629]
            if curr_infer_length == 0:
                samples = x
            else:
                samples = torch.cat([samples, x], 1)
        
        return samples

    def loss(self, X, Y, with_energy=False, reduce=True):
        #X is a categorical distribution and Y is the target (one hot encoding).
        X = X.permute(0, 2, 1)
        #X.size() = [256, 19, 629] after permuting; Y.size() = [256, 19]
        #256 = batchsize, 629 = number classes, 19 = number of dihedrals
        loss_value = self.loss_function(X, Y) #nn.CrossEntropyLoss automatically applies softmax
        return loss_value
    
    def exact_log_likelihood(self, sample):
        """
        Calculate exact log likelihood of a given sample. 
        """
        #dihedrals should be represented as indicies already
        output = self.forward(decoder_input=sample)
        #dist = dist.permute(0, 2, 1)
        distribution = dist.Categorical(output)
        prob = torch.sum(distribution.log_prob(sample), dim=1)
        return loss

    def energy_loss(self, batch_size, sum=False):
        """
        Currently doesn't work. Supposed to be DL(q || p). 
        Can't figure out what the beta term is supposed to be.
        """
        ba, dihedral = model.sample(batch_size)
        dihedral_true = utilities.index_to_number(dihedral)
        q_s = (-model.exact_log_likelihood(ba=ba, dihedrals=dihedral))
        E_s = dataset_train.compute_potential_energy_for_ic(
            utilities.rebuild(
                torch.cat([ba, dihedral_true], dim=1).cpu().detach()
            ), unitless=False
        )
        return torch.sum(torch.add(torch.tensor(E_s).to(self.device), q_s.to(self.device)))