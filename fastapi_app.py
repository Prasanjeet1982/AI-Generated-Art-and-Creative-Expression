from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from io import BytesIO
from starlette.responses import StreamingResponse
from app import ai_generator, pg_gan_model

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generate_art/")
async def generate_art(model: tf.keras.models.Model = Depends(pg_gan_model.load_pg_gan_model)):
    generated_image = ai_generator.generate_ai_art(model)
    img_byte_array = BytesIO()
    generated_image.save(img_byte_array, format="PNG")
    return StreamingResponse(io.BytesIO(img_byte_array.getvalue()), media_type="image/png")
