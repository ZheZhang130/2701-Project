"""Portable probability-average wrapper for multiclass candidate exports."""
import hashlib,json
from pathlib import Path
import numpy as np
from .experiments import CandidateClassifier
from .advanced import AdvancedClassifier

class EnsembleClassifier:
    def __init__(self,bundle,device='mps'):
        bundle=Path(bundle);self.metadata=json.loads((bundle/'bundle.json').read_text())
        if self.metadata['schema_version']!=4:raise ValueError('Unsupported ensemble bundle')
        self.members=[]
        for member in self.metadata['members']:
            path=(bundle/member['path']).resolve()
            if not path.is_relative_to(bundle.resolve()):raise ValueError('Member path escapes bundle')
            if hashlib.sha256((path/'bundle.json').read_bytes()).hexdigest()!=member['metadata_sha256']:
                raise ValueError('Member metadata checksum mismatch')
            metadata=json.loads((path/'bundle.json').read_text())
            loader=AdvancedClassifier if metadata.get('schema_version')==3 else CandidateClassifier
            classifier=loader(path,device)
            if classifier.metadata['candidate']['objective']!='multiclass':raise ValueError('Ten-class members required')
            self.members.append(classifier)
        if len(self.members)<2:raise ValueError('At least two members required')
    def predict(self,images):
        if not images:raise ValueError('Images required')
        p=np.mean([np.array([r['probabilities'] for r in member.predict(images)]) for member in self.members],axis=0)
        return [dict(probabilities=row.tolist(),predicted_class=int(row.argmax()),frame_score=float(1-row[0]),
                     model_version=self.metadata['model_version']) for row in p]
