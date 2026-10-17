"""Model catalog for SHIVAM.MS AI ENHANCE V3.
Only real downloadable weights are listed here. Model availability can change upstream.
"""
MODEL_CATALOG = {
    "Real-ESRGAN x4plus (general)": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "file": "RealESRGAN_x4plus.pth",
        "description": "General-purpose x4 image restoration model.",
        "scale": 4,
    },
    "Real-ESRGAN x4plus Anime 6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "description": "Anime/illustration-focused x4 model.",
        "scale": 4,
    },
    "Real-ESRGAN x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "file": "RealESRGAN_x2plus.pth",
        "description": "General-purpose x2 model; lighter output scale.",
        "scale": 2,
    },
    "Real-ESRNet x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "file": "RealESRNet_x4plus.pth",
        "description": "x4 restoration model with a smoother reconstruction tendency.",
        "scale": 4,
    },
    "Official ESRGAN SRx4 DF2KOST": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        "file": "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        "description": "Official ESRGAN x4 weights from the Real-ESRGAN model zoo.",
        "scale": 4,
    },
}
