from fastapi import Depends
from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from fastapi import File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lib.images import PolygonDrawer
from lib.images import image_to_img_src
from lib.images import open_image
from lib.models import Reader
from lib.models import get_model

import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"))
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def get_form(request: Request) -> Response:
    return templates.TemplateResponse(request, "upload.html")


@app.post("/upload", response_class=HTMLResponse)
def upload(
    request: Request,
    file: UploadFile = File(...),
    model: Reader = Depends(get_model, use_cache=True)
):
    return infer_model(file, request, model)


@app.post("/", response_class=HTMLResponse)
def infer_model(
    file: UploadFile,
    request: Request,
    model: Reader = Depends(get_model, use_cache=True),
) -> Response:
    ctx: dict = {}
    try:
        image = open_image(file.file)
        draw = PolygonDrawer.from_image(image)
        words = []
        for coords, word, accuracy in model.readtext(np.array(image)):
            draw.highlight_word(coords, word)
            cropped_word_image = draw.crop(coords)
            words.append(
                {
                    "image": image_to_img_src(cropped_word_image),
                    "word": word,
                    "accuracy": accuracy,
                }
            )
        highlighted_image = draw.get_highlighted_image()
        ctx.update(
            image=image_to_img_src(highlighted_image),
            words=words,
        )
    except Exception as err:
        print(err)
        ctx.update(error=str(err))
    return templates.TemplateResponse(request, "info.html", ctx)
