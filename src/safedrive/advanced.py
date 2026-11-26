"""Additional development candidates; existing model and monitoring bundles stay unchanged."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models,transforms
from PIL import Image
from .experiments import transform,outputs
from .model import select_device

REGIONS=((0.,0.,1.,1.),(.05,0.,.65,.75),(.30,.25,1.,1.))

class RegionTransform:
    """Fixed, label-independent regions specified after training-only geometry inspection."""
    def __init__(self,size,training=False):
        self.transform=transform(size,False)
        self.augment=transforms.ColorJitter(.2,.2,.1) if training else None
    def __call__(self,image):
        if not isinstance(image,Image.Image) or image.mode!='RGB':raise ValueError('RGB image required')
        if self.augment:image=self.augment(image)
        w,h=image.size
        return torch.stack([self.transform(image.crop((round(x*w),round(y*h),round(r*w),round(b*h))))
                            for x,y,r,b in REGIONS])

def advanced_transform(candidate,training=False):
    if candidate['architecture']=='resnet18_regions':return RegionTransform(candidate['size'],training)
    return transform(candidate['size'],training)

class RegionBackbone(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        self.net=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc=nn.Identity()
    def forward(self,x):
        if x.ndim!=5 or x.shape[1]!=3:raise ValueError('Expected three image views')
        b,v,c,h,w=x.shape
        return self.net(x.reshape(b*v,c,h,w)).reshape(b,v*512)

class AdvancedNet(nn.Module):
    def __init__(self,candidate,pretrained=False,dino_source=None):
        super().__init__();self.candidate=dict(candidate)
        arch=candidate['architecture']
        if arch=='resnet50':
            self.backbone=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            width=self.backbone.fc.in_features;self.backbone.fc=nn.Identity()
        elif arch=='convnext_tiny':
            self.backbone=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            width=self.backbone.classifier[2].in_features;self.backbone.classifier[2]=nn.Identity()
        elif arch=='resnet18_regions':
            self.backbone=RegionBackbone(pretrained);width=1536
        elif arch=='dinov2_vits14':
            os.environ['XFORMERS_DISABLED']='1'
            if dino_source is None:
                root=Path(__file__).resolve().parents[2]
                pin=json.loads((root/'artifacts/advanced/dino_source.json').read_text())
                dino_source=root/pin['path']
            self.backbone=torch.hub.load(str(dino_source),'dinov2_vits14',source='local',pretrained=pretrained)
            width=384
        else:raise ValueError('Unsupported advanced candidate')
        self.head=nn.Linear(width,10)
    def forward(self,x):return self.head(self.backbone(x))
    def set_trainable(self,epoch):
        arch=self.candidate['architecture']
        for name,p in self.backbone.named_parameters():
            if self.candidate.get('frozen'):enabled=False
            elif arch=='resnet50':enabled=name.startswith(('layer3.','layer4.'))
            elif arch=='resnet18_regions':enabled=name.startswith(('net.layer3.','net.layer4.'))
            elif arch=='convnext_tiny':enabled=name.startswith(('features.5.','features.6.','features.7.','classifier.'))
            elif arch=='dinov2_vits14':enabled=name.startswith(('blocks.10.','blocks.11.','norm.'))
            else:raise ValueError(arch)
            p.requires_grad=enabled
        for p in self.head.parameters():p.requires_grad=True

def average_predictions(predictions):
    """Align by IDs and reject mismatched labels/drivers; average probabilities, not logits."""
    if not predictions:raise ValueError('Predictions required')
    first=predictions[0];ids=first['image_ids'].tolist()
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate image IDs')
    arrays=[]
    for pred in predictions:
        other=pred['image_ids'].tolist()
        if len(other)!=len(set(other)) or set(ids)!=set(other):raise ValueError('Image ID mismatch')
        index={name:i for i,name in enumerate(other)};order=[index[name] for name in ids]
        if not np.array_equal(first['labels'],pred['labels'][order]) or not np.array_equal(first['subjects'],pred['subjects'][order]):
            raise ValueError('Label/driver mismatch')
        p,_=outputs(pred['logits'][order],'multiclass');arrays.append(p)
    p=np.mean(arrays,axis=0)
    return dict(labels=first['labels'],logits=np.log(np.clip(p,1e-15,1)),
                image_ids=first['image_ids'],subjects=first['subjects'])

class AdvancedClassifier:
    def __init__(self,bundle,device='mps'):
        bundle=Path(bundle);self.metadata=json.loads((bundle/'bundle.json').read_text())
        if self.metadata['schema_version']!=3:raise ValueError('Unsupported advanced bundle')
        if hashlib.sha256((bundle/'model.pt').read_bytes()).hexdigest()!=self.metadata['weights_sha256']:
            raise ValueError('Weights checksum mismatch')
        for name,digest in self.metadata.get('vendor_sha256',{}).items():
            path=(bundle/name).resolve()
            if not path.is_relative_to(bundle.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Vendor checksum mismatch')
        self.device=select_device(device)
        self.model=AdvancedNet(self.metadata['candidate'],dino_source=bundle/'vendor' if (bundle/'vendor').exists() else None).to(self.device)
        self.model.load_state_dict(torch.load(bundle/'model.pt',map_location='cpu',weights_only=True));self.model.eval()
        self.transform=advanced_transform(self.metadata['candidate'])
    @torch.inference_mode()
    def predict(self,images):
        if not images:raise ValueError('Images required')
        logits=self.model(torch.stack([self.transform(im) for im in images]).to(self.device)).cpu().numpy()
        p,s=outputs(logits,'multiclass',self.metadata.get('temperatures'))
        return [dict(probabilities=p[i].tolist(),predicted_class=int(p[i].argmax()),frame_score=float(s[i]),
                     model_version=self.metadata['model_version']) for i in range(len(images))]
