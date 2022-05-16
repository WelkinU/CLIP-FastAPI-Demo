# CLIP FastAPI Server

This repo contains a FastAPI front end interface to CLIP (Contrastive Language-Image Pre-Training) from OpenAI [[Github]](https://github.com/openai/CLIP) [[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020)

This is intended to be a user-friendly demo of the algorithm capabilities. Just enter the categories into the textarea and drag + drop images into the drop zone. Then classification percentages appear on the right.

![image](https://user-images.githubusercontent.com/47000850/166162679-ce2bbe8a-47db-470e-bc9c-6090faa95e87.png)

## Running FastAPI Web Server

In the command prompt, use `python clip_server.py` to start the web server. By default it runs on `localhost:8000` and reloads whenever the code is updated + saved. Check to see that the server is running properly by navigating to `localhost:8000` in your web browser.

| Argument | Default | Description |
| --- | --- | :--- |
| `--host` | localhost |Update host parameter |
| `--port` | 8000 | Update port parameter |
| `--model` | ViT-B/32 |Set CLIP model to use. Current available models: RN50 RN101 RN50x4 RN50x16 RN50x64 ViT-B/32 ViT-B/16 ViT-L/14 ViT-L/14@336px. See [OpenAI's CLIP Blog Post](https://openai.com/blog/clip/) for model metrics / GFLOPs.
| `--device` |  | CUDA device, i.e. 0 or 0,1,2,3 or cpu |
| `--prod` | False | Convienience flag that sets host to local IPv4 address. Does not reload the server upon code updates/saves. |

## Dependencies

Can use `pip install -r requirements.txt` or...

### CLIP Dependancies

1. Make sure you have PyTorch / torchvision installed.
1. Install small additional dependancies with `pip install ftfy regex tqdm`
1. Install CLIP from the OpenAI Github repo with `pip install git+https://github.com/openai/CLIP.git`

### FastAPI + Pillow

1. FastAPI: `pip install fastapi`
1. Pillow (PIL): `pip install pillow`
