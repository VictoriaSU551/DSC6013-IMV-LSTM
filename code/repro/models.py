"""Independent PyTorch implementation of Guo et al. equations (1)-(13).

Not copied from the unlicensed upstream snapshots. Output heads are per-variable;
variable scorer is shared. Temporal context excludes the final hidden state.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def mixture_statistics(mu, sigma, logits, y):
    """Eq.8/10/13; detached EM responsibilities give the exact NLL gradient.

    The global I term in Eq.9 is constant w.r.t. theta for a frozen E-step;
    its closed-form M-step is posterior.mean(0), exported by the trainer.
    """
    log_prior=F.log_softmax(logits,dim=-1)
    log_density=-.5*((y[:,None]-mu)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)
    log_joint=log_density+log_prior
    q=F.softmax(log_joint,dim=-1).detach()
    return {'prediction':(log_prior.exp()*mu).sum(-1),
            'posterior':q,'nll':-torch.logsumexp(log_joint,dim=-1).mean(),
            'em_loss':-(q*log_joint).sum(-1).mean()}


class IMV(nn.Module):
    def __init__(self,n,d,kind='tensor',dropout=0.,sigma_floor=.001):
        super().__init__()
        if n<1 or d<1 or kind not in ('full','tensor'):
            raise ValueError('positive n/d and full or tensor required')
        self.n,self.d,self.kind,self.sigma_floor=n,d,kind,sigma_floor
        if kind=='tensor':
            self.rec_w=nn.Parameter(torch.empty(n,d,4*d))
            self.in_w=nn.Parameter(torch.empty(n,4*d))
            self.rec_bias=nn.Parameter(torch.zeros(n,4*d))
        else:
            self.rec_w=nn.Parameter(torch.empty(n,d,d))
            self.in_w=nn.Parameter(torch.empty(n,d))
            self.rec_bias=nn.Parameter(torch.zeros(n,d))
            self.gates=nn.Linear(n+n*d,3*n*d)
        nn.init.uniform_(self.rec_w,-1/math.sqrt(d),1/math.sqrt(d))
        nn.init.uniform_(self.in_w,-1/math.sqrt(d),1/math.sqrt(d))
        if kind=='tensor':
            with torch.no_grad(): self.rec_bias[:,2*d:3*d].fill_(1.)
        else:
            with torch.no_grad(): self.gates.bias[n*d:2*n*d].fill_(1.)
        self.temp_w=nn.Parameter(torch.empty(n,d)); nn.init.normal_(self.temp_w,std=.1)
        self.temp_bias=nn.Parameter(torch.zeros(n))
        self.prior_head=nn.Linear(2*d,1)
        self.mean_w=nn.Parameter(torch.empty(n,2*d)); nn.init.normal_(self.mean_w,std=.1)
        self.mean_bias=nn.Parameter(torch.zeros(n))
        self.scale_w=nn.Parameter(torch.empty(n,2*d)); nn.init.normal_(self.scale_w,std=.02)
        self.scale_bias=nn.Parameter(torch.full((n,),-1.5))
        self.dropout=nn.Dropout(dropout)

    def recurrent_parameter_count(self):
        total=self.rec_w.numel()+self.in_w.numel()+self.rec_bias.numel()
        return total+(sum(p.numel() for p in self.gates.parameters()) if self.kind=='full' else 0)

    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.n or x.shape[1]<2:
            raise ValueError('x must be [batch,T>=2,n]')
        h=x.new_zeros((len(x),self.n,self.d)); c=torch.zeros_like(h)
        history=[]
        for t in range(x.shape[1]):
            z=torch.einsum('bnd,ndh->bnh',h,self.rec_w)+x[:,t,:,None]*self.in_w+self.rec_bias
            if self.kind=='tensor':
                j,i,f,o=z.chunk(4,-1)
                j=j.tanh(); i=i.sigmoid(); f=f.sigmoid(); o=o.sigmoid()
            else:
                j=z.tanh()
                i,f,o=self.gates(torch.cat([x[:,t],h.reshape(len(x),-1)],-1)).reshape(len(x),3,self.n,self.d).unbind(1)
                i=i.sigmoid(); f=f.sigmoid(); o=o.sigmoid()
            c=f*c+i*j
            h=o*c.tanh(); history.append(h)
        hidden=torch.stack(history,dim=1)
        previous=hidden[:,:-1]
        alpha=torch.softmax(torch.tanh((previous*self.temp_w).sum(-1)+self.temp_bias),dim=1)
        context=(alpha[...,None]*previous).sum(1)
        summary=self.dropout(torch.cat([context,hidden[:,-1]],-1))
        logits=self.prior_head(summary).squeeze(-1).tanh()
        prior=logits.softmax(-1)
        mu=(summary*self.mean_w).sum(-1)+self.mean_bias
        sigma=F.softplus((summary*self.scale_w).sum(-1)+self.scale_bias)+self.sigma_floor
        return {'prediction':(prior*mu).sum(-1),'mu':mu,'sigma':sigma,'logits':logits,
                'prior':prior,'importance':prior,'alpha':alpha,'hidden':hidden}


class LSTMRegressor(nn.Module):
    def __init__(self,n,h,dropout):
        super().__init__(); self.n=n
        self.rnn=nn.LSTM(n,h,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(h,1))
    def forward(self,x):
        hidden,_=self.rnn(x)
        return {'prediction':self.head(hidden[:,-1]).squeeze(-1),
                'importance':x.new_full((len(x),self.n),1/self.n)}


class RETAIN(nn.Module):
    """Regression adaptation of reverse-time attention; rank via |contribution|."""
    def __init__(self,n,h,dropout):
        super().__init__(); self.embed=nn.Linear(n,h,bias=False)
        self.rnn_a=nn.GRU(h,h,batch_first=True); self.rnn_b=nn.GRU(h,h,batch_first=True)
        self.att_a=nn.Linear(h,1); self.att_b=nn.Linear(h,h)
        self.out=nn.Linear(h,1); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        emb=self.drop(self.embed(x)); reversed_emb=emb.flip(1)
        a,_=self.rnn_a(reversed_emb); b,_=self.rnn_b(reversed_emb)
        alpha=self.att_a(a).softmax(1).flip(1)
        beta=self.att_b(b).tanh().flip(1)
        context=(alpha*beta*emb).sum(1)
        coeff=torch.einsum('bth,h,hn->btn',alpha*beta,self.out.weight[0],self.embed.weight)
        contrib=(coeff*x).abs().sum(1)
        importance=(contrib+1e-12)/(contrib.sum(-1,keepdim=True)+1e-12*x.shape[-1])
        return {'prediction':self.out(context).squeeze(-1),'importance':importance,
                'alpha':alpha.squeeze(-1),'contribution':coeff*x}


class DUAL(nn.Module):
    """DA-RNN input attention + temporal decoder, using only the past window.

    The autoregressive decoder receives the observed target history separately.
    Encoder receives registered exogenous variables; target is excluded from encoder.
    If target is absent after selection the decoder history remains available;
    this protocol difference is explicitly reported.
    """
    def __init__(self,n,h,depth,target,dropout):
        super().__init__(); self.h,self.n,self.target=h,n,target
        self.encoder_indices=[i for i in range(n) if i!=target]
        if not self.encoder_indices: raise ValueError('DUAL needs at least one exogenous encoder input')
        self.enc=nn.LSTMCell(len(self.encoder_indices),h); self.input_h=nn.Linear(2*h,depth)
        self.input_x=nn.Linear(depth,depth,bias=False); self.input_v=nn.Linear(depth,1,bias=False)
        self.dec=nn.LSTMCell(1,h); self.time_h=nn.Linear(2*h,h)
        self.time_x=nn.Linear(h,h,bias=False); self.time_v=nn.Linear(h,1,bias=False)
        self.y_context=nn.Linear(h+1,1)
        self.out=nn.Sequential(nn.Dropout(dropout),nn.Linear(2*h,1))
    def forward(self,x,target_history=None):
        batch,T,_=x.shape
        if target_history is None:
            if self.target is None: raise ValueError('target history required')
            target_history=x[:,:,self.target]
        h=x.new_zeros(batch,self.h); c=torch.zeros_like(h); enc=[]; alphas=[]
        encoder_x=x[:,:,self.encoder_indices]
        key=self.input_x(encoder_x.transpose(1,2))
        for t in range(T):
            scores=self.input_v(torch.tanh(self.input_h(torch.cat([h,c],-1))[:,None]+key)).squeeze(-1)
            a=scores.softmax(-1)
            h,c=self.enc(a*encoder_x[:,t],(h,c)); enc.append(h); alphas.append(a)
        encoded=torch.stack(enc,1); key_time=self.time_x(encoded)
        h=x.new_zeros(batch,self.h); c=torch.zeros_like(h); temporal=[]
        for t in range(T):
            scores=self.time_v(torch.tanh(self.time_h(torch.cat([h,c],-1))[:,None]+key_time)).squeeze(-1)
            b=scores.softmax(-1); context=(b[...,None]*encoded).sum(1)
            y_in=self.y_context(torch.cat([context,target_history[:,t,None]],-1))
            h,c=self.dec(y_in,(h,c)); temporal.append(b)
        importance=x.new_zeros(batch,self.n)
        importance[:,self.encoder_indices]=torch.stack(alphas,1).mean(1)
        return {'prediction':self.out(torch.cat([h,context],-1)).squeeze(-1),
                'importance':importance,'alpha':torch.stack(temporal,1)}


def build_model(kind,n,units,depth=10,target=None,dropout=0.,sigma_floor=.001):
    if kind in ('full','tensor'): return IMV(n,units,kind,dropout,sigma_floor)
    if kind=='lstm': return LSTMRegressor(n,units,dropout)
    if kind=='retain': return RETAIN(n,units,dropout)
    if kind=='dual': return DUAL(n,units,depth,target,dropout)
    raise ValueError(kind)
