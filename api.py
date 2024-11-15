import soundfile as sf
import torch
import tqdm
from cached_path import cached_path

from model import DiT, UNetT
from model.utils import save_spectrogram

from model.utils_infer import load_vocoder, load_model, infer_process, remove_silence_for_generated_wav
from model.utils import seed_everything
import random
import sys

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
import json
import uuid
import os
import uvicorn
import traceback
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles


class F5TTS:
    def __init__(
        self,
        model_type="F5-TTS",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=True,
        local_path=None,
        device=None,
    ):
        # Initialize parameters
        self.final_wave = None
        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.seed = -1

        # Set device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load models
        self.load_vocoder_model(local_path)
        self.load_ema_model(model_type, ckpt_file, vocab_file, ode_method, use_ema)

    def load_vocoder_model(self, local_path):
        self.vocos = load_vocoder(local_path is not None, local_path, self.device)

    def load_ema_model(self, model_type, ckpt_file, vocab_file, ode_method, use_ema):
        if model_type == "F5-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            model_cls = DiT
        elif model_type == "E2-TTS":
            if not ckpt_file:
                ckpt_file = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            model_cls = UNetT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file, ode_method, use_ema, self.device)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spect, file_spect):
        save_spectrogram(spect, file_spect)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        sway_sampling_coef=-1,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        fix_duration=None,
        remove_silence=False,
        file_wave=None,
        file_spect=None,
        seed=-1,
    ):
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed
        wav, sr, spect = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence)

        if file_spect is not None:
            self.export_spectrogram(spect, file_spect)

        return wav, sr, spect

COUNT = 0
app = FastAPI()


# Directory for uploaded files and JSON storage
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "out"
DATA_FILE = "ref_info.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reference_data = json.load(f)
else:
    reference_data = {}

app.mount("/out", StaticFiles(directory="out"), name="out")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SynthesisRequest(BaseModel):
    gen_text: str = Field("要合成的文本内容", description="要合成的文本内容")
    ref_id: str = Field("参考音频的ID，从上传接口响应体中获取", description="参考音频的ID，从上传接口响应体中获取")  # Use ref_id to fetch stored audio and text
    seed: int = Field(-1, description="随机种子，-1表示不使用随机种子")
    api_key: str = Field("密钥", description="API Key")


class SynthesisRequest2(BaseModel):
    gen_text: str = Field("要合成的文本内容", description="要合成的文本内容")
    ref_audio_path: str = Field(..., description="服务器中，参考音频的相对路径")
    ref_text: str = Field("参考音频的文本内容", description="参考文本内容")
    seed: int = Field(-1, description="随机种子，-1表示不使用随机种子")
    api_key: str = Field("密钥", description="API Key")


@app.post("/tts", description="合成音频，通过上传接口返回的ref_id选定参考音频信息进行合成")
async def _tts(request: SynthesisRequest):
    global COUNT

    try:
        if AUTH:
            if request.api_key != API_KEY:
                logger.warning("API Key错误")
                return {
                    "code": 401,
                    "success": False,
                    "msg": "API Key错误",
                }
            
        # Retrieve reference data using ref_id
        ref_data = reference_data.get(request.ref_id)
        if not ref_data:
            raise HTTPException(status_code=404, detail="Reference ID not found")

        # Extract the reference audio path and text
        ref_file_path = ref_data["ref_audio_path"]
        ref_text = ref_data["ref_text"]

        # Limit the number of iterations to 1000
        COUNT = (COUNT + 1) % 1000

        # Generate output file paths
        wave_file = os.path.join(OUTPUT_DIR, f"out_{COUNT}.wav")
        spect_file = os.path.join(OUTPUT_DIR, f"out_{COUNT}.png")

        # Run the synthesis
        wav, sr, spect = f5tts.infer(
            ref_file=ref_file_path,
            ref_text=ref_text,
            gen_text=request.gen_text,
            file_wave=wave_file,
            file_spect=spect_file,
            seed=request.seed,
        )

        if wav is None:
            logger.error("合成失败")
            return {
                "code": 500,
                "success": False,
                "msg": "合成失败",
            }
        else:
            logger.info(f"合成成功, 输出文件到: {wave_file}")
            return {
                "code": 0,
                "success": True,
                "out_audio_path": wave_file,
                "msg": "合成成功"
            }
    except Exception as e:
        logger.error(traceback.format_exc())
        return {
            "code": 500,
            "success": False,
            "msg": str(e),
        }


# 新的 tts2 接口，直接使用相对路径和参考文本
@app.post("/tts2", description="直接使用服务器程序相对路径的参考音频和参考文本进行音频合成")
async def _tts2(request: SynthesisRequest2):
    global COUNT

    try:
        if AUTH:
            if request.api_key != API_KEY:
                logger.warning("API Key错误")
                return {
                    "code": 401,
                    "success": False,
                    "msg": "API Key错误",
                }

        # 使用传入的相对路径和文本
        ref_file_path = request.ref_audio_path
        ref_text = request.ref_text

        # 检查参考文件是否存在
        if not os.path.exists(ref_file_path):
            raise HTTPException(status_code=404, detail="参考音频文件不存在")

        # 限制迭代次数
        COUNT = (COUNT + 1) % 1000

        # 生成输出文件路径
        wave_file = os.path.join(OUTPUT_DIR, f"out2_{COUNT}.wav")
        spect_file = os.path.join(OUTPUT_DIR, f"out2_{COUNT}.png")

        # 运行合成
        wav, sr, spect = f5tts.infer(
            ref_file=ref_file_path,
            ref_text=ref_text,
            gen_text=request.gen_text,
            file_wave=wave_file,
            file_spect=spect_file,
            seed=request.seed,
        )

        if wav is None:
            logger.error("合成失败")
            return {
                "code": 500,
                "success": False,
                "msg": "合成失败",
            }
        else:
            logger.info(f"合成成功, 输出文件到: {wave_file}")
            return {
                "code": 0,
                "success": True,
                "out_audio_path": wave_file,
                "msg": "合成成功"
            }
    except Exception as e:
        logger.error(traceback.format_exc())
        return {
            "code": 500,
            "success": False,
            "msg": str(e),
        }

@app.post("/upload_ref", description="上传参考音频文件和参考文本到服务端")
async def _upload_ref(
    ref_file: UploadFile = File(..., description="参考音频文件"),
    ref_text: str = Form(..., description="参考文本"),
    api_key: str = Form(..., description="API Key"),
):
    try:
        if AUTH:
            if api_key != API_KEY:
                logger.warning("API Key错误")
                return {
                    "code": 401,
                    "success": False,
                    "msg": "API Key错误",
                }

        # Save uploaded files with a unique identifier
        ref_id = str(uuid.uuid4())
        ref_audio_path = os.path.join(UPLOAD_DIR, f"{ref_id}_{ref_file.filename}")

        # Write file and store reference text
        with open(ref_audio_path, "wb") as f:
            f.write(await ref_file.read())

        # Save data to JSON
        reference_data[ref_id] = {
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(reference_data, f)

        logger.info(f"上传成功，ref_id:{ref_id}, ref_audio_path:{ref_audio_path}, ref_text:{ref_text}")

        return {
            "code": 0,
            "success": True,
            "ref_id": ref_id, 
            "ref_audio_path": ref_audio_path, 
            "ref_text": ref_text,
            "msg": "上传成功"
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        return {
            "code": 500,
            "success": False,
            "msg": str(e),
        }

if __name__ == "__main__":
    logger.add("log.txt", level="INFO", rotation="100 MB")

    f5tts = F5TTS()

    # 是否验证api_key
    AUTH = True
    API_KEY = "20242024"

    uvicorn.run(app, host="0.0.0.0", port=9000)
