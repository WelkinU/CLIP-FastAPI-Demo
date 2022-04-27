# CLIP FastAPI Server

This repo contains a FastAPI front end interface to CLIP (Contrastive Language-Image Pre-Training) from OpenAI [[Github]](https://github.com/openai/CLIP) [[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020)

This is intended to be a user friendly demo of the algorithm capabilities.


## Dependencies

### CLIP Dependancies

1. Make sure you have PyTorch / torchvision installed
1. Install small additional dependancies with `pip install ftfy regex tqdm`
1. Install CLIP from the OpenAI Github repo with `pip install git+https://github.com/openai/CLIP.git`

### FastAPI + Pillow

1. FastAPI: `pip install fastapi`
1. Pillow (PIL): `pip install pillow`

### Running FastAPI Web Server

In the command prompt, use `python clip_server.py` to start the web server. By default it runs on `localhost:8000` and reloads whenever the code is updated + saved. Use `--prod` for production deployment. Check to see that the server is running properly by navigating to `localhost:8000` in your web browser.

The server autodetects if PyTorch is installed, if not, the neural network inference endpoint is disabled.

| Argument | Description |
| --- | :--- |
| `--host` | Update host parameter. Default `localhost` |
| `--port` | Update port parameter. Default `8000` |
| `--prod` | Convienience feature for production deployment. Defaults host to local IPv4 address and port 8000. Does not reload the server upon code updates/saves. |


