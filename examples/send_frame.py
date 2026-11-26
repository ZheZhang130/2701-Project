"""Send a JPEG/PNG frame without third-party client dependencies."""
import argparse,json,os,urllib.parse,urllib.request
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('image',type=Path)
parser.add_argument('--url',default='http://127.0.0.1:7860')
parser.add_argument('--model',default='resnet18')
parser.add_argument('--session',default='demo-trip')
parser.add_argument('--frame',default='frame-0')
parser.add_argument('--timestamp-ms',type=int,default=0)
parser.add_argument('--stateless',action='store_true')
args=parser.parse_args()
route='predict' if args.stateless else 'frames?'+urllib.parse.urlencode(dict(
    session_id=args.session,frame_id=args.frame,timestamp_ms=args.timestamp_ms))
request=urllib.request.Request(f'{args.url}/v1/models/{args.model}/{route}',data=args.image.read_bytes(),
    headers={'X-API-Key':os.environ['SAFEDRIVE_API_KEY'],
             'Content-Type':'image/png' if args.image.suffix.lower()=='.png' else 'image/jpeg'})
with urllib.request.urlopen(request,timeout=60) as response:print(json.dumps(json.load(response),indent=2))
