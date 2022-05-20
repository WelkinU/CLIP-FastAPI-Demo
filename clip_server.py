from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

import torch
import clip
from PIL import Image
from io import BytesIO
import os.path

import argparse

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

@app.get("/")
async def detect(request: Request):
    '''Returns GUI HTML / Javascript template for the Detect Run web form
    '''
    return templates.TemplateResponse('detect_form.html', {
            "request": request,
        })

@app.post('/predict_demo')
async def predict(file: UploadFile = File(...), category_list: str = Form(...)):
    ''' Intended to be called from detect_form.html
    '''
    category_list = category_list.replace('[','').replace(']','')
    category_list = [s.strip() for s in category_list.split(',')]
    text = clip.tokenize(category_list).to(device)
    image = preprocess(Image.open(BytesIO(await file.read()))).unsqueeze(0).to(device)

    with torch.no_grad():
        #image_features = model.encode_image(image)
        #text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(category_list)
    print(str(list(probs[0])))

    return JSONResponse({'probs':str(list(probs[0]))},
            status_code=200)

def get_ip():
    ''' Gets local IPv4 address. From https://stackoverflow.com/a/28950776'''

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def select_device(device=''):
    ''' Modified from https://github.com/ultralytics/yolov5/blob/4870064629e76d9d387578c562a292cc680fa05f/utils/torch_utils.py#L52
     device = 'cpu' or '0' or '0,1,2,3' '''
    s = f''  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    return torch.device('cuda:0' if cuda else 'cpu')

#This code block has to be outside the if __name__ == '__main__' block to have globals set properly in worker processes
parser = argparse.ArgumentParser()
parser.add_argument('--host', help='IP Address where the server is hosted', type=str, default='localhost')
parser.add_argument('--port', help='Port number where the server is hosted', type=int, default='8000')
parser.add_argument('--model', help=f'CLIP models - All models: {clip.available_models()}', type=str, default="ViT-B/32")
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--prod', action='store_true', 
                help='Convenience function for setting to local IPv4 as host, and reload set to False.')
args = parser.parse_args()

device = select_device(args.device)
model, preprocess = clip.load(args.model, device=device)

if __name__ == '__main__':
    import uvicorn

    current_filename = os.path.splitext(os.path.basename(__file__))[0]
    if args.prod:
        import socket #for getting IPv4 address for get_ip() function
        uvicorn.run(f"{current_filename}:app", host=get_ip(), port=args.port, reload=False)
    else:
        uvicorn.run(f"{current_filename}:app", host=args.host, port=args.port, reload=True)
