# 2020-10-02 
# This is the script which I practice implementing of Transformer model
# Reference ) https://www.youtube.com/watch?v=U0s0f995w14&t=1800s

import torch 
import torch.nn as nn 

# first, make necessary blocks 

class SelfAttention(nn.Module): 
    def  __init__(self, embed_size, heads): 
        # heads : how many splits from embed_size 
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size 
        self.heads = heads 
        self.head_dim = embed_size // heads   # integer division 

        assert (self.head_dim*heads  == embed_size) , "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim,  self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) # not necessary to write embed_size, but  make clear 
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)  # for concatenation 

        def forward(self, values, keys, query, mask): 
            N = query.shape[0] # how many examples we send at same time  
            value_len, key_len, query_len =  values.shape[1], key.shape[1], query.shape[1] 
            # it depends on source sentence len and target sentence len 

            # Split embedding into self.heads pieces    
            # befor spliting, it was single dimension, just embed size 
            # by splitting, divide it to self.heads, self.head_dim
            values = values.reshape(N, value_len, self.heads, self.head_dim) # last two parts are important 
            keys  = keys.reshape(N, key_len, self.heads, self.head_dim)
            queries = query.reshape(N, query_len, self.heads, self.head_dim)

            energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
            # queries shape : (N, query_len, heads, heads_dim) 
            # keys shape : (N, key_len, heads, heads_dim)
            # energy shape : (N, heads, query_len, key_len)
            # instead of you can use torch.bmm (batch matrix multiplication)

            if mask is not None: 
                energy = energy.masked_fill(mask == 0, float("-1e20")) # to go minus infinite value 
                # th) if a value of mask is 0 ,change the value to "-1e20" to set it as very small value 
                # th) via softmax, it would be almost zero to diminish attention wights 
            # triangular matrix 
            # replacing minus infinity just for numerical under overflowed anything like that 
            # we wnat it to set it to a very small value 

            attention = torch.softmax(energy / (self.embed_size **(1/2)), dim=3) 
            # attention is applied to key part as source values, so that write dim=3 
            out = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(
                N, query_len, self.heads, self.heads_dim
            )
            # th) I just thouhgt it was simple matix multiplication 
            # th) But when we multiply them, we have to check each dimension of input matrix 
            # attention shape : (N, heads, query_len, key_len)
            # values shape : (N, value_len, heads, heads_dim)
            # after einsum (N, query_len, heads, heads_dim) then flatten last two dims for concatenating
            # Note that size between key_len and value_len is same 
            out = self.fc_out(out)
            return out 

# th) Why did I notice it nowdays which 'Linear' means linear trasnformation 
# th) It is manipulication of space to find best space to find appropriate featrue space !!  
# th)  nn.Linear means  
class TrasnformerBlock(nn.Module): 
    def __init__(self, embed_size, heads, dropout, forward_expansion): 
        super(TrasnformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask): 
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out  

class Encoder(nn.Module): 
    def __init__(
        self,
        src_vocab_size, 
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
        ):  
        super(Encoder, self).__init__()
        self.embed_size = embed_size 
        self.device = device 
        self.word_embedding = nn.Embedding(src_vocal_size, embed_size) 
        self.position_embedding = nn.Embdeding(max_length, embed_size)
        # th)learnable embedding table
        # th) so,, it means just  make set of embedding vectors and then make train to recognize each word? 
        # nn.Embedding(num_embeddings: int, embedding_dim: int)
        # num_embeddings : size of the dictionary of embeddings / embedding_dim : size of each embedding vector

        self.layers = nn.ModuleList(
            [
                TrasnformerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask): 
        N, seq_len = x.shape 
        positions = torch.arange(0,seq_len).expand(N, seq_len).to(self.device)
        # torch.arange 
        # Returns a 1-D tensor of size (end-start) / (step)  
        # with values from the interval [start, end) taken with common difference step beginning from start
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers: 
            out = layer(out, out, out, mask)  # when Trasnsformer.forward(value, key, query, mask)
        return out 

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TrasnformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask): 
        attention = self.attention(x, x, x,trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out 

class Decoder(nn.Module): 
    def __init__(self,
                trg_vocab_size,
                embed_size,
                num_leyers,
                heads,
                forward_expansion,
                dropout, 
                device,
                max_length): 
        super(Decoder, self).__init__()
        self.device = device 
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_leyers)]   
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

        def forward(self, x, enc_out, src_mask, trg_mask):
            N, seq_len = x.shape
            positions = torch.arange(0, seq_len).expand(N,seq_len).to(self.device)
            x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

            for layer in self.layers : 
                x = layer(x, enc_out, src_mask, trg_mask)
            
            out = self.fc_out(x)
            return out 

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size, 
            trg_vocab_size, 
            src_pad_idx,
            trg_pad_idx,
            embed_size=256, 
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            devide="cuda", 
            max_length=100
    ):
        super(Transformer,self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.docoder = Decoder(trg_vocab_size, embed_size, num_leyers, heads, forward_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx 
        self.device = device 

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsuqeeze(1).unsqueeze(2) # th) 0 or 1 ? 
        # (N, 1, 1, src_len)
        return src_mask.to(device)

    def make_trg_mask(self, trg): 
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg): 
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out 


if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    


        
