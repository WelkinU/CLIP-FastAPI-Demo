from fastapi import FastAPI, Form, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

import torch
import clip
from PIL import Image
from io import BytesIO
import os.path

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
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

if __name__ == '__main__':
    import uvicorn
    import argparse

    available_clip_models = clip.available_models()
    default_clip_model = "ViT-B/32"

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='IP Address where the server is hosted', type=str, default='localhost')
    parser.add_argument('--port', help='Port number where the server is hosted', type=int, default='8000')
    parser.add_argument('--model', help=f'CLIP models - All models: {available_clip_models}', 
                        type=str, default=default_clip_model, choices=available_clip_models)
    parser.add_argument('--prod', action='store_true', help='Default to local IPv4 as host, and reload set to False.')
    args = parser.parse_args()

    current_filename = os.path.splitext(os.path.basename(__file__))[0]
    if args.prod:
        import socket #for getting IPv4 address for get_ip() function
        uvicorn.run(f"{current_filename}:app", host=get_ip(), port=args.port, reload=False)
    else:
        uvicorn.run(f"{current_filename}:app", host=args.host, port=args.port, reload=True)

    model, preprocess = clip.load(args.model, device=device)

else:
    model, preprocess = clip.load("ViT-B/32", device=device) #in case this is somehow imported

