"""Experimental models and metrics; v1 production-facing classifier stays intact."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps
from safedrive.metrics import classification_metrics
from safedrive.model import select_device


class ResizePad:
    def __init__(self, size): self.size = size
    def __call__(self, image):
        if not isinstance(image, Image.Image) or image.mode != 'RGB':
            raise ValueError('Expected RGB PIL image')
        return ImageOps.pad(image, (self.size, self.size), method=Image.Resampling.BILINEAR,
                            color=(124,116,104))


def transform(size, training=False):
    steps=[ResizePad(size)]
    if training:
        steps += [transforms.ColorJitter(.2,.2,.1),
                  transforms.RandomApply([transforms.GaussianBlur(3,sigma=(.1,1.))],p=.15)]
    return transforms.Compose(steps+[transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])


class ExperimentNet(nn.Module):
    def __init__(self, candidate, pretrained=False):
        super().__init__()
        self.candidate=dict(candidate)
        if candidate['architecture']=='resnet18':
            self.backbone=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            width=self.backbone.fc.in_features
            self.backbone.fc=nn.Identity()
        elif candidate['architecture']=='efficientnet_b0':
            self.backbone=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            width=self.backbone.classifier[1].in_features
            self.backbone.classifier=nn.Identity()
        else:
            raise ValueError('Unsupported architecture')
        objective=candidate['objective']
        if objective not in ('multiclass','binary','joint'): raise ValueError('Invalid objective')
        self.head=nn.Linear(width, {'multiclass':10,'binary':1,'joint':11}[objective])
    def forward(self,x): return self.head(self.backbone(x))

    def set_trainable(self, epoch):
        gradual=self.candidate['gradual']
        for name,p in self.backbone.named_parameters():
            if self.candidate['architecture']=='efficientnet_b0':
                enabled=name.startswith(('features.6.','features.7.','features.8.'))
            else:
                enabled=name.startswith(('layer3.','layer4.'))
                if gradual and epoch>=3: enabled |= name.startswith('layer2.')
                if gradual and epoch>=5: enabled=True
            p.requires_grad=enabled
        for p in self.head.parameters(): p.requires_grad=True


def loss_for(logits, y, objective, pos_weight):
    binary=(y!=0).float()
    if objective=='binary':
        return nn.functional.binary_cross_entropy_with_logits(logits[:,0],binary,pos_weight=pos_weight)
    ce=nn.functional.cross_entropy(logits[:,:10],y)
    if objective=='joint':
        ce=ce+.5*nn.functional.binary_cross_entropy_with_logits(logits[:,10],binary,pos_weight=pos_weight)
    return ce


def outputs(logits, objective, temperatures=None):
    logits=np.asarray(logits,dtype=np.float64)
    temperatures=temperatures or {'class':1.,'binary':1.}
    probabilities=None
    if objective!='binary':
        z=logits[:,:10]/temperatures['class'];z-=z.max(1,keepdims=True)
        probabilities=np.exp(z);probabilities/=probabilities.sum(1,keepdims=True)
    if objective=='multiclass': score=1-probabilities[:,0]
    else:
        z=logits[:,0 if objective=='binary' else 10]/temperatures['binary']
        score=1/(1+np.exp(-np.clip(z,-700,700)))
    return probabilities, score


def binary_metrics(y, score, threshold=None):
    y=np.asarray(y)!=0;score=np.asarray(score,dtype=np.float64)
    if len(y)==0 or len(y)!=len(score) or not np.isfinite(score).all() or ((score<0)|(score>1)).any():
        raise ValueError('Invalid binary inputs')
    if not y.any() or y.all(): raise ValueError('Both binary classes required')
    order=np.argsort(-score,kind='stable'); sy=y[order];ss=score[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    tp=np.r_[0,np.cumsum(sy)[ends]];fp=np.r_[0,np.cumsum(~sy)[ends]]
    tpr=tp/y.sum();fpr=fp/(~y).sum();thresholds=np.r_[1.0000000001,ss[ends]]
    allowed=np.flatnonzero(fpr<=.05)
    k=allowed[np.argmax(tpr[allowed])]
    clipped=np.clip(score,1e-12,1-1e-12)
    result={'roc_auc':float(np.trapezoid(tpr,fpr)),
        'binary_log_loss':float(-(y*np.log(clipped)+(~y)*np.log(1-clipped)).mean()),
        'brier':float(((score-y)**2).mean()),'safe_frames':int((~y).sum()),'non_safe_frames':int(y.sum()),
        'diagnostic_recall_at_fpr5':float(tpr[k]),'diagnostic_fpr':float(fpr[k]),
        'diagnostic_threshold':float(thresholds[k])}
    if threshold is not None:
        flag=score>=threshold
        result['fixed_threshold']=float(threshold)
        result['safe_fpr']=float(flag[~y].mean())
        result['non_safe_recall']=float(flag[y].mean())
    return result


def summarize(y, logits, subjects, objective, temperatures=None, threshold=None):
    p,s=outputs(logits,objective,temperatures)
    result={'binary':binary_metrics(y,s,threshold)}
    if p is not None: result['multiclass']=classification_metrics(y,p)
    result['per_driver']={}
    subjects=np.asarray(subjects)
    for d in sorted(set(subjects)):
        mask=subjects==d
        item={'images':int(mask.sum()),'binary':binary_metrics(np.asarray(y)[mask],s[mask],threshold)}
        if p is not None: item['multiclass']=classification_metrics(np.asarray(y)[mask],p[mask])
        result['per_driver'][str(d)]=item
    return result


def fit_temperatures(y,logits,objective):
    # Tiny post-processing optimization on CPU; no backbone work or training images.
    result={'class':1.,'binary':1.}
    grid=np.unique(np.r_[np.geomspace(.25,8.,161),1.])
    if objective!='binary':
        losses=[]
        for t in grid:
            p,_=outputs(logits,objective,{'class':float(t),'binary':1.})
            losses.append(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean())
        result['class']=float(grid[np.argmin(losses)])
    if objective!='multiclass':
        losses=[]
        for t in grid:
            _,s=outputs(logits,objective,{'class':1.,'binary':float(t)})
            losses.append(binary_metrics(y,s)['binary_log_loss'])
        result['binary']=float(grid[np.argmin(losses)])
    return result


class CandidateClassifier:
    def __init__(self,bundle,device='mps'):
        bundle=Path(bundle);self.metadata=json.loads((bundle/'bundle.json').read_text())
        if self.metadata['schema_version']!=2: raise ValueError('Unsupported candidate bundle')
        if hashlib.sha256((bundle/'model.pt').read_bytes()).hexdigest()!=self.metadata['weights_sha256']:
            raise ValueError('Model checksum mismatch')
        self.device=select_device(device)
        self.model=ExperimentNet(self.metadata['candidate']).to(self.device)
        self.model.load_state_dict(torch.load(bundle/'model.pt',map_location='cpu',weights_only=True))
        self.model.eval();self.transform=transform(self.metadata['candidate']['size'])
    @torch.inference_mode()
    def predict(self,images):
        if not images: raise ValueError('Images required')
        logits=self.model(torch.stack([self.transform(im) for im in images]).to(self.device)).cpu().numpy()
        p,s=outputs(logits,self.metadata['candidate']['objective'],self.metadata.get('temperatures'))
        return [{'probabilities':None if p is None else p[i].tolist(),
                 'predicted_class':None if p is None else int(p[i].argmax()),
                 'frame_score':float(s[i]),'model_version':self.metadata['model_version']} for i in range(len(images))]
