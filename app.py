"""SHIVAM.MS AI ENHANCE V7 — resume path and queue selection fix.
Loads one selected, real pretrained model on demand; it does not waste GPU time loading unused models.
"""
import os, sys, shutil, subprocess, tempfile, time, urllib.request, threading

# Detect the hosting runtime before importing Hugging Face ZeroGPU. A hosted HF Space
# cannot see or borrow a separate Colab GPU; only the Colab-launched process can use it.
IS_COLAB_RUNTIME = (
    os.getenv("SHIVAM_RUNTIME", "").lower() == "colab"
    or "google.colab" in sys.modules
    or (os.path.isdir("/content") and not os.getenv("SPACE_ID"))
)
if IS_COLAB_RUNTIME:
    def gpu_required(fn):
        return fn  # Colab manages CUDA directly; no Hugging Face ZeroGPU decorator.
else:
    try:
        import spaces  # Hugging Face ZeroGPU is only used inside an HF Space.
        gpu_required = spaces.GPU(duration=300)
    except Exception:
        def gpu_required(fn):
            return fn
from pathlib import Path
import cv2
import numpy as np
import gradio as gr
from models import MODEL_CATALOG
from effects import apply_effects, EFFECT_PRESETS

APP_TITLE = "SHIVAM.MS AI ENHANCE V7 GRAPHICS"
MODEL_DIR = Path(os.getenv("SHIVAM_MODEL_DIR", str(Path.home() / ".cache" / "shivam_ms_models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE = {}
MODEL_LOCK = threading.Lock()
IS_HF_ZERO_GPU = (not IS_COLAB_RUNTIME) and bool(os.getenv("SPACE_ID")) and os.getenv("SPACE_HARDWARE", "").lower().startswith("zero")

# QUNQE PROCESS SAVE: durable job records, source uploads, frame checkpoints and persistent outputs.
from datetime import datetime, timezone, timedelta
import uuid, json, queue, hashlib, secrets, re
JOB_ROOT = Path(os.getenv("SHIVAM_JOB_DIR", str(Path.home() / ".shivam_ms_jobs")))
JOB_ROOT.mkdir(parents=True, exist_ok=True)
JOB_INPUTS = JOB_ROOT / "inputs"; JOB_INPUTS.mkdir(exist_ok=True)
JOB_FRAMES = JOB_ROOT / "frames"; JOB_FRAMES.mkdir(exist_ok=True)
JOB_OUTPUTS = Path(os.getenv("SHIVAM_OUTPUT_DIR", str(JOB_ROOT / "downloads")))
JOB_OUTPUTS.mkdir(parents=True, exist_ok=True)
JOB_DB = JOB_ROOT / "jobs.json"
JOBS_LOCK = threading.RLock()
JOB_QUEUE = queue.Queue()
JOB_ACTIVE = {"id": None}
RETENTION_DAYS = None  # Account-owned files are not auto-expired.
# Separate app-only storage (not Google Drive). Colab local disk is ephemeral across runtime resets.
APP_STORAGE = Path(os.getenv("SHIVAM_APP_STORAGE", str(JOB_ROOT / "account_storage")))
APP_STORAGE.mkdir(parents=True, exist_ok=True)
APP_STORAGE_LIMIT = int(1.5 * 1024**3)
ACCOUNT_DB = APP_STORAGE / "accounts.json"

def _now(): return datetime.now(timezone.utc).isoformat()
def _load_jobs():
    try:
        data = json.loads(JOB_DB.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception: return {}
def _save_jobs(jobs):
    tmp = JOB_DB.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(jobs, indent=2), encoding='utf-8')
    tmp.replace(JOB_DB)
def _update_job(job_id, **updates):
    with JOBS_LOCK:
        jobs = _load_jobs()
        if job_id in jobs:
            jobs[job_id].update(updates); jobs[job_id]['updated_at'] = _now(); _save_jobs(jobs)

def _cleanup_expired():
    # Account-owned originals, checkpoints, leftovers and completed outputs are durable
    # user files; never delete them automatically just because a retention timer elapsed.
    return

def _job_progress(job_id):
    class P:
        def __call__(self, fraction, desc=None, **kwargs):
            try: pct = max(0.0, min(99.0, float(fraction)*100))
            except Exception: pct = 0.0
            _update_job(job_id, percent=round(pct, 1), stage=str(desc or 'Processing'))
            # Keep the recovery video at a stable path. Renaming it on every
            # progress tick invalidates the dropdown's selected value and can
            # make Resume unable to match the saved file to its queue record.
            # Frame progress is stored in jobs.json; checkpoints are the resume source.
    return P()

def _run_job(job_id):
    with JOBS_LOCK:
        jobs = _load_jobs(); job = jobs.get(job_id)
        if not job: return
        if JOB_ACTIVE['id'] is not None: return
        JOB_ACTIVE['id'] = job_id
        job['status']='running'; job['stage']='Starting/resuming from saved frame checkpoints'; job['updated_at']=_now(); _save_jobs(jobs)
    try:
        params = job['params']
        result, report = process_video(
            job['input_path'], params['model_name'], params['scale_name'], params['preset'],
            params['denoise'], params['sharpen'], params['structure'], params['clarity'],
            params['glow'], params['saturation'], params['contrast'], params['brightness'],
            params['grain'], params['max_seconds'], progress=_job_progress(job_id),
            checkpoint_dir=str((_account_folder(job.get("owner")) / "QUNQE Processes" / "Resume Checkpoints" / job_id) if job.get("owner") else (JOB_FRAMES / job_id)), output_dir=str((_account_folder(job.get("owner")) / "Processed Videos") if job.get("owner") else JOB_OUTPUTS), job_id=job_id
        )
        # Keep a separate local app-storage copy, only while the signed-in owner's 1.5 GB quota permits it.
        with JOBS_LOCK:
            current = _load_jobs().get(job_id, {})
        owner = current.get("owner")
        try:
            named = Path(result).with_name(_safe_stem(current.get('name', 'Enhanced video')) + Path(result).suffix)
            if named != Path(result):
                if named.exists(): named = named.with_name(f'{named.stem}_{int(time.time())}{named.suffix}')
                Path(result).rename(named); result = str(named)
        except Exception: pass
        _update_job(job_id, status='completed', percent=100.0, stage='Completed — saved in Processed Videos', output_path=result,
                    app_storage_copy=result if owner else None,
                    app_storage_note=None if owner else 'Finished video remains in configured output storage.',
                    report=report, completed_at=_now())
    except Exception as exc:
        _update_job(job_id, status='paused', stage=f'Paused after interruption/error: {type(exc).__name__}: {exc}', error=str(exc))
    finally:
        JOB_ACTIVE['id'] = None

def _worker():
    while True:
        jid = JOB_QUEUE.get()
        try: _run_job(jid)
        except Exception as exc: _update_job(jid, status='paused', stage=f'Worker stopped: {exc}', error=str(exc))
        finally: JOB_QUEUE.task_done()

def _restore_jobs():
    _cleanup_expired()
    with JOBS_LOCK:
        jobs = _load_jobs(); queue_ids=[]; changed=False
        for jid, job in jobs.items():
            if job.get('status') in ('running','queued'):
                job['status']='paused'; job['stage']=f"Runtime restarted; saved progress {job.get('percent',0)}%. Press Resume to continue."; job['updated_at']=_now(); changed=True
        if changed: _save_jobs(jobs)
threading.Thread(target=_worker, daemon=True, name='qunqe-job-worker').start()
_restore_jobs()

def _load_accounts():
    try:
        value=json.loads(ACCOUNT_DB.read_text(encoding="utf-8"))
        return value if isinstance(value,dict) else {}
    except Exception: return {}
def _save_accounts(accounts):
    tmp=ACCOUNT_DB.with_suffix(".tmp"); tmp.write_text(json.dumps(accounts,indent=2),encoding="utf-8"); tmp.replace(ACCOUNT_DB)
def _pw_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 180000).hex()
def _account_action(action, username, password, current_user):
    username=(username or "").strip()
    if action == "Log out": return None, "Logged out.", gr.update(choices=sorted(_load_accounts()), value=None)
    if action == "Switch account":
        if not current_user: return None, "Log in first to switch account.", gr.update(choices=sorted(_load_accounts()), value=None)
        return None, "Logged out. Select another app account and log in.", gr.update(choices=sorted(_load_accounts()), value=None)
    if not username or not password: raise gr.Error("Enter an app username and password.")
    with JOBS_LOCK:
        accounts=_load_accounts()
        if action == "Create account":
            if len(username)<3 or len(username)>32 or not username.replace("_", "").isalnum(): raise gr.Error("Username must be 3–32 letters, numbers, or underscores.")
            if len(password)<8: raise gr.Error("Use a password with at least 8 characters.")
            if username.lower() in {k.lower() for k in accounts}: raise gr.Error("That app account already exists.")
            salt=secrets.token_hex(16); accounts[username]={"salt":salt,"password_hash":_pw_hash(password,salt),"created_at":_now()}; _save_accounts(accounts)
            return username, f"Signed in as **{username}**. Account created.", gr.update(choices=sorted(accounts), value=username)
        if action == "Log in":
            key=next((k for k in accounts if k.lower()==username.lower()),None)
            if not key or not secrets.compare_digest(accounts[key]["password_hash"],_pw_hash(password,accounts[key]["salt"])): raise gr.Error("Incorrect app username or password.")
            return key, f"Signed in as **{key}**.", gr.update(choices=sorted(accounts), value=key)
    raise gr.Error("Choose Create account, Log in, Log out, or Switch account.")
def _account_folder(user):
    folder = APP_STORAGE / hashlib.sha256(user.encode()).hexdigest()[:20]
    for sub in ("QUNQE Processes/Original Videos", "QUNQE Processes/Leftover Videos", "QUNQE Processes/Resume Checkpoints", "Processed Videos", "Downloads"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    return folder

def _dir_bytes(folder):
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) if folder.exists() else 0

def _safe_stem(value, fallback="Video process"):
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", (value or "").strip()).strip(" ._- ")
    return (value[:90] or fallback)

def _video_files(folder):
    folder=Path(folder); valid={".mp4",".mov",".mkv",".avi",".webm",".m4v"}
    if not folder.exists(): return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid), key=lambda p:p.stat().st_mtime, reverse=True)

def _rename_file(path, stem):
    path=Path(path)
    if not path.is_file(): return str(path)
    target=path.with_name(_safe_stem(stem)+path.suffix)
    if target == path: return str(path)
    if target.exists(): target=path.with_name(f"{_safe_stem(stem)}_{int(time.time())}{path.suffix}")
    path.rename(target); return str(target)

def _copy_into_app_storage(user, source, category, preferred_name=None):
    """Copies a local app-side copy if the per-account 1.5 GB quota permits it."""
    if not user or not source or not Path(source).is_file(): return None
    folder = _account_folder(user)
    dest_dir = folder / category
    dest = dest_dir / (preferred_name or Path(source).name)
    if dest.exists() and dest.resolve() == Path(source).resolve(): return str(dest)
    current = _dir_bytes(folder)
    incoming = Path(source).stat().st_size
    previous = dest.stat().st_size if dest.exists() else 0
    if current - previous + incoming > APP_STORAGE_LIMIT:
        return None
    shutil.copy2(source, dest)
    return str(dest)

def _storage_view(user):
    if not user:
        return {"account":"Not logged in", "used_MB":0, "limit_MB":round(APP_STORAGE_LIMIT/1024**2), "free_MB":round(APP_STORAGE_LIMIT/1024**2), "files":[], "note":"Log in to view this account's separate app storage."}
    folder = _account_folder(user)
    used = 0; files = []; by_category = {"QUNQE Processes":0,"Processed Videos":0,"Downloads":0}
    for f in folder.rglob("*"):
        if f.is_file():
            size=f.stat().st_size; used+=size
            try: cat=f.relative_to(folder).parts[0]
            except Exception: cat="Other"
            if cat in by_category: by_category[cat]+=size
            elif cat == "QUNQE Processes": by_category[cat]+=size
            files.append({"name":str(f.relative_to(folder)),"size_MB":round(size/1024**2,2)})
    # Show this account's job states too, while file bytes remain under the local 1.5 GB cap.
    with JOBS_LOCK: jobs=_load_jobs()
    owned=[{"id":j.get("id"),"name":j.get("name"),"status":j.get("status"),"percent":j.get("percent",0)} for j in jobs.values() if j.get("owner")==user]
    return {"account":user,"used_MB":round(used/1024**2,2),"limit_MB":round(APP_STORAGE_LIMIT/1024**2),"free_MB":round(max(0,APP_STORAGE_LIMIT-used)/1024**2,2),"categories_MB":{k:round(v/1024**2,2) for k,v in by_category.items()},"files":files,"QUNQE_processes":owned,"note":"Account folders and job records persist when SHIVAM_JOB_DIR and SHIVAM_APP_STORAGE point to Google Drive. On other hosts, persistence depends on that host’s storage policy."}

def get_ffmpeg():
    p = shutil.which("ffmpeg")
    if p: return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("FFmpeg unavailable. Check requirements.txt: " + str(e))


def download_model(name, progress):
    item = MODEL_CATALOG[name]
    dest = MODEL_DIR / item["file"]
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    progress(0.04, desc=f"Downloading real model: {name}")
    try:
        req = urllib.request.Request(item["url"], headers={"User-Agent": "SHIVAM-MS-AI-ENHANCE/3.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", "0") or 0); got = 0
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk: break
                f.write(chunk); got += len(chunk)
                if total:
                    progress(min(0.20, 0.04 + 0.16 * got / total), desc=f"Downloading weights… {got//(1024*1024)} MB")
        if tmp.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded weights file is unexpectedly small; upstream link may have changed.")
        tmp.replace(dest)
    finally:
        if tmp.exists(): tmp.unlink(missing_ok=True)
    return dest


def load_model(name, progress):
    with MODEL_LOCK:
        if name in MODEL_CACHE:
            return MODEL_CACHE[name]
        import torch
        from spandrel import ModelLoader
        path = download_model(name, progress)
        progress(0.23, desc=f"Loading selected model {name} into memory…")
        descriptor = ModelLoader().load_from_file(str(path))
        model = descriptor.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        MODEL_CACHE[name] = (model, device)
        return MODEL_CACHE[name]


def run_model(frame, model, device, output_scale):
    import torch
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model(tensor)
        if hasattr(out, "image"): out = out.image
        elif isinstance(out, (tuple, list)): out = out[0]
    arr = out.squeeze(0).detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    result = cv2.cvtColor((arr * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    target = (frame.shape[1] * output_scale, frame.shape[0] * output_scale)
    if result.shape[1] != target[0] or result.shape[0] != target[1]:
        result = cv2.resize(result, target, interpolation=cv2.INTER_LANCZOS4)
    del tensor, out
    return result


@gpu_required
def process_video(input_path, model_name, scale_name, preset, denoise, sharpen, structure, clarity,
                  glow, saturation, contrast, brightness, grain, max_seconds, progress=gr.Progress(),
                  checkpoint_dir=None, output_dir=None, job_id=None):
    started = time.perf_counter()
    if not input_path or not os.path.isfile(input_path): raise gr.Error("Upload a source video first.")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise gr.Error("Cannot read the video. Try MP4, MOV, or AVI.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if total > 0 else 0
    if width <= 0 or height <= 0:
        cap.release(); raise gr.Error("Video reports invalid dimensions.")
    if max_seconds and duration > max_seconds:
        cap.release(); raise gr.Error(f"Clip is {duration:.1f}s, above the selected {max_seconds}s safety limit. Increase it if your runtime has enough memory/time.")
    scale = {"Original size": 1, "2× upscale": 2, "4× upscale": 4}[scale_name]
    if width * height * scale * scale > 35_000_000:
        cap.release(); raise gr.Error("This output would be very large. Choose Original size or 2× for this input resolution.")
    raw = tempfile.NamedTemporaryFile(suffix="_video_only.mp4", delete=False).name
    # In Colab, save finished videos directly to Google Drive when mounted.
    output_dir = Path(output_dir or os.getenv("SHIVAM_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "shivam_ms_outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final = str(output_dir / f"SHIVAM_MS_ENHANCED_{job_id or stamp}.mp4")
    writer = None; count = 0; model = device = None
    checkpoints = Path(checkpoint_dir) if checkpoint_dir else None
    if checkpoints: checkpoints.mkdir(parents=True, exist_ok=True)
    try:
        # Resume from the contiguous, actually-existing checkpoint sequence. Progress metadata alone is never trusted.
        if checkpoints:
            while (checkpoints / f"frame_{count+1:08d}.jpg").is_file(): count += 1
            if total and count > total:
                for f in checkpoints.glob('frame_*.jpg'): f.unlink(missing_ok=True)
                count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        model, device = load_model(model_name, progress)
        progress(0.27 + (0.56 * count / total if total else 0), desc=f"Restored {count} saved frames" if count else "Starting frame-by-frame enhancement…")
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_no = count + 1
            try:
                frame = run_model(frame, model, device, scale)
                frame = apply_effects(frame, denoise, sharpen, structure, clarity, glow,
                                      saturation, contrast, brightness, grain, preset)
            except Exception as exc:
                raise RuntimeError(f"Frame {frame_no} failed ({type(exc).__name__}): {exc}") from exc
            if checkpoints:
                cp = checkpoints / f"frame_{frame_no:08d}.jpg"
                if not cv2.imwrite(str(cp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98]):
                    raise RuntimeError(f"Could not save checkpoint frame {frame_no}; check persistent storage space.")
            else:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened(): raise RuntimeError("Could not create temporary video writer.")
                writer.write(frame)
            count += 1
            if total: progress(min(0.84, 0.27 + 0.56 * count / total), desc=f"Enhancing frame {count}/{total} — checkpoint saved" if checkpoints else f"Enhancing frame {count}/{total}")
            if count % 4 == 0:
                try:
                    import torch
                    if device == "cuda": torch.cuda.empty_cache()
                except Exception: pass
        cap.release()
        if writer: writer.release(); writer = None
        if count == 0: raise RuntimeError("No decodable frames found in the uploaded video.")
        if checkpoints:
            first = cv2.imread(str(checkpoints / "frame_00000001.jpg"))
            if first is None: raise RuntimeError("Checkpoint frame 1 is missing or unreadable.")
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened(): raise RuntimeError("Could not create checkpoint video writer.")
            for n in range(1, count + 1):
                f = cv2.imread(str(checkpoints / f"frame_{n:08d}.jpg"))
                if f is None: raise RuntimeError(f"Checkpoint frame {n} is missing or unreadable.")
                writer.write(f)
            writer.release(); writer = None
        ffmpeg = get_ffmpeg()
        cmd = [ffmpeg, "-y", "-i", raw, "-i", input_path, "-map", "0:v:0", "-map", "1:a?",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "19", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", final]
        progress(0.88, desc="Restoring source audio and packaging MP4…")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # No artificial 5-minute packaging timeout; Colab/runtime limits still apply.
        if result.returncode: raise RuntimeError("FFmpeg packaging failed: " + result.stderr[-900:])
        if not os.path.isfile(final) or os.path.getsize(final) < 1024: raise RuntimeError("Output file was not created correctly.")
        elapsed = time.perf_counter() - started
        report = (f"Model: {model_name}\nPreset: {preset}; denoise {denoise}/10; sharpen {sharpen}%; structure {structure}%; clarity {clarity}%; glow {glow}%.\n"
                  f"Frames: {count}; FPS preserved: {fps:.3f}; duration ≈ {count/fps:.2f}s; output {width*scale}×{height*scale}; runtime {elapsed:.1f}s.\n"
                  f"Inference device: {device.upper()} (CUDA GPU is used only when PyTorch reports it available).\n"
                  "Original audio was copied through AAC when present. This is frame-wise restoration; it does not create 60 FPS interpolation or true HDR mastering.")
        progress(1.0, desc="Finished")
        return final, report
    except gr.Error: raise
    except Exception as exc:
        raise gr.Error(f"Enhancement failed: {exc}")
    finally:
        cap.release()
        if writer:
            try: writer.release()
            except Exception: pass
        for p in (raw,):
            try: os.remove(p)
            except OSError: pass
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass



def process_video_router(input_path, model_name, scale_name, preset, denoise, sharpen, structure, clarity,
                        glow, saturation, contrast, brightness, grain, max_seconds, colab_url):
    """Route processing to a manually started Colab Gradio share URL when supplied.

    The temporary Gradio share URL is a bearer secret: keep it private and replace it
    whenever Colab restarts. If blank, process locally on the Space.
    """
    url = (colab_url or "").strip().rstrip("/")
    if not url:
        return process_video(input_path, model_name, scale_name, preset, denoise, sharpen, structure, clarity,
                             glow, saturation, contrast, brightness, grain, max_seconds)
    if not input_path:
        raise gr.Error("Upload a source video first.")
    try:
        from gradio_client import Client, handle_file
        client = Client(url, verbose=False)
        remote_input = handle_file(input_path)
        result = client.predict(
            remote_input, model_name, scale_name, preset, denoise, sharpen, structure, clarity,
            glow, saturation, contrast, brightness, grain, max_seconds, api_name="/process_video"
        )
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RuntimeError(f"Unexpected Colab response: {type(result).__name__}")
        return result[0], "Processed on manually started Colab GPU.\n\n" + str(result[1])
    except Exception as exc:
        raise gr.Error("Could not reach/process on Colab. Start the Colab notebook, wait for its Gradio share URL, paste that URL here, and try again. Details: " + str(exc))



# Startup preload: cache every real model's weight file to Drive/disk, but do not
# hold all neural networks in GPU memory. Only the selected model is loaded for inference.
BOOT_STARTED = time.time()
BOOT_LOCK = threading.Lock()
BOOT = {"ready": False, "done": 0, "total": len(MODEL_CATALOG), "status": "Preparing runtime…", "detail": "Checking dependencies and GPU", "error": ""}
BOOT_MIN_SECONDS = 90

def _boot_progress(fraction, desc="Preparing model weights"):
    with BOOT_LOCK:
        current = int(BOOT.get("done", 0))
        total = max(1, int(BOOT.get("total", 1)))
        BOOT["status"] = desc
        BOOT["fraction"] = min(0.98, (current + max(0.0, min(1.0, float(fraction))) / total))

def _startup_preload():
    """Fast startup diagnostics; download/load only the model the user selects."""
    try:
        import torch
        with BOOT_LOCK:
            BOOT["status"] = "Checking GPU and video pipeline"
            if torch.cuda.is_available():
                BOOT["detail"] = f"CUDA GPU detected: {torch.cuda.get_device_name(0)}. The selected model will download and load when enhancement starts."
            else:
                BOOT["detail"] = "CUDA is not available to PyTorch. The app can run on CPU, but choose a GPU runtime and restart if you want GPU acceleration."
            BOOT["fraction"] = 0.35
        test_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        apply_effects(test_frame, 0, 0, 0, 0, 0, 100, 100, 0, 0, EFFECT_PRESETS[0])
        with BOOT_LOCK:
            BOOT["done"] = len(MODEL_CATALOG)
            BOOT["fraction"] = 1.0
            BOOT["status"] = "Runtime checks complete"
            BOOT["detail"] += " Model weights are loaded on demand to avoid long startup stalls."
            BOOT["ready"] = True
    except Exception as exc:
        with BOOT_LOCK:
            BOOT["error"] = f"{type(exc).__name__}: {exc}"
            BOOT["status"] = "Runtime setup needs attention"
            BOOT["detail"] = "See the displayed error, then restart the runtime if needed."
            BOOT["ready"] = True

threading.Thread(target=_startup_preload, daemon=True, name="shivam-startup-preload").start()

CSS = r"""
:root { --sm-accent:#7c5cff; }
.gradio-container { background: radial-gradient(ellipse at top,#24203c 0%,#10121a 55%,#0b0d12 100%) !important; }
#hero { padding:24px 18px; border-radius:20px; background:linear-gradient(115deg,#171923,#2b1f50,#142a38); border:1px solid #51417d; overflow:hidden; }
#hero h1 { letter-spacing:1px; font-size:clamp(25px,4vw,42px); margin:0; }
#hero p { color:#c8c8dc; margin:8px 0 0; }
#loadstrip { margin-top:16px; height:5px; border-radius:8px; overflow:hidden; background:#303242; }
#loadstrip:after { content:''; display:block; height:100%; width:100%; background:linear-gradient(90deg,#7c5cff,#36d6d0); transform-origin:left; animation:loadReveal .9s ease-out both; }
@keyframes loadReveal { from{transform:scaleX(0);opacity:.5} to{transform:scaleX(1);opacity:1} }
#boot-card { max-width:760px; margin:6vh auto; padding:clamp(22px,5vw,42px); border:1px solid #51417d; border-radius:26px; background:linear-gradient(145deg,rgba(24,25,40,.97),rgba(36,25,67,.96),rgba(15,35,47,.97)); box-shadow:0 20px 80px rgba(0,0,0,.32); text-align:center; animation:cardEnter .85s cubic-bezier(.2,.8,.2,1) both; }
@keyframes cardEnter { from{opacity:0;transform:translateY(16px) scale(.97)} to{opacity:1;transform:translateY(0) scale(1)} }
#boot-logo { width:72px;height:72px;border-radius:22px;margin:0 auto 18px;display:grid;place-items:center;font-size:26px;font-weight:900;color:white;background:linear-gradient(135deg,#7c5cff,#36d6d0);box-shadow:0 0 35px #7c5cff55;animation:logoEnter .85s cubic-bezier(.2,.8,.2,1) both; }
@keyframes logoEnter { from{opacity:0;transform:scale(.72) translateY(10px)} to{opacity:1;transform:scale(1) translateY(0)} }
#boot-title {font-size:clamp(24px,5vw,38px);font-weight:850;letter-spacing:.6px;color:#fff}
#boot-sub {color:#c8c8dc;margin:10px 0 24px;line-height:1.5}
#boot-meter {height:8px;border-radius:12px;background:#292b3b;overflow:hidden;margin:20px 0 12px}
#boot-fill {height:100%;border-radius:12px;background:linear-gradient(90deg,#7c5cff,#36d6d0);width:0%;transition:width .6s ease;box-shadow:0 0 18px #36d6d077}
#boot-pulse {display:none}

.boot-stat {display:inline-block;margin:5px;padding:10px 14px;border:1px solid #45435f;border-radius:12px;background:#171927;color:#f1f0ff;font-variant-numeric:tabular-nums}
.boot-note {font-size:12px;color:#a9abc2;line-height:1.55;margin-top:18px}
"""

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css=CSS) as demo:
    boot_screen = gr.Column(visible=True, elem_id="boot-screen")
    with boot_screen:
        boot_html = gr.HTML(value="""<div id='boot-card'><div id='boot-logo'>SM</div><div id='boot-title'>SHIVAM.MS <span style='color:#72e5e0'>AI ENHANCE</span></div><div id='boot-sub'>Preparing your enhancement workspace<br>Checking GPU • preparing video pipeline • loading selected model on demand</div><div id='boot-pulse'></div><div id='boot-meter'><div id='boot-fill' style='width:2%'></div></div><div class='boot-stat'>Elapsed: <b id='boot-elapsed'>0:00</b></div><div class='boot-stat'>Minimum screen time left: <b id='boot-left'>1:30</b></div><div style='margin-top:16px;color:#d8d8ed'>Starting model preparation…</div><div class='boot-note'>Please keep this tab open while setup runs. Model weights are cached to Drive; neural networks are loaded into GPU memory only when needed, to avoid exhausting VRAM.</div></div>""")
        boot_status = gr.Markdown("**Status:** Starting…")
        boot_timer = gr.Timer(value=1.0, active=True)
    with gr.Column(visible=False, elem_id="main-menu") as main_menu:
        account_session = gr.State(None)
        gr.Markdown("### APP ACCOUNT · Sign in to save and manage your QUNQE progress")
        with gr.Row():
            account_name = gr.Textbox(label="App username", placeholder="Create or choose an app account")
            account_password = gr.Textbox(label="Password", type="password")
            account_action = gr.Dropdown(["Create account","Log in","Log out","Switch account"], value="Log in", label="Account action")
            account_button = gr.Button("Continue", variant="primary")
        account_picker = gr.Dropdown(label="Saved app accounts (this app server)", choices=[])
        account_list_refresh = gr.Button("Refresh account list")
        account_status = gr.Markdown("Not logged in. Create an app account or log in. This cannot list Android/Google accounts on your phone; browsers do not expose those accounts to apps.")
        account_button.click(_account_action, inputs=[account_action,account_name,account_password,account_session], outputs=[account_session,account_status,account_picker])
        account_list_refresh.click(lambda: gr.update(choices=sorted(_load_accounts())), inputs=[], outputs=[account_picker])
        account_picker.change(lambda choice: (choice or "", "Select **Log in** and enter that account's password to switch safely."), inputs=[account_picker], outputs=[account_name,account_status])
        gr.HTML("""<div id='hero'><h1>SHIVAM.MS AI ENHANCE <span style='color:#72e5e0'>V7</span></h1><p>AI restoration • CUDA-aware GPU processing • checkpoint resume • audio-preserving export</p><div id='loadstrip'></div><p style='font-size:12px'>Choose a real pretrained model and tune the finishing effects below.</p></div>""")
        gr.Markdown("### 1 · Video input and model")
        with gr.Row():
            video_in = gr.Video(label="Upload video", sources=["upload"])
            video_out = gr.Video(label="Enhanced output")
        with gr.Row():
            model_choice = gr.Dropdown(choices=list(MODEL_CATALOG.keys()), value="Real-ESRGAN x4plus (general)", label="Real pretrained AI model")
            scale_choice = gr.Radio(["Original size", "2× upscale", "4× upscale"], value="Original size", label="Output resolution")
            preset = gr.Dropdown(EFFECT_PRESETS, value="Clean Natural", label="Finishing preset (not a separate AI model)")
        with gr.Accordion("2 · AI detail controls", open=True):
            with gr.Row():
                denoise = gr.Slider(0, 10, value=2, step=1, label="Denoise strength")
                sharpen = gr.Slider(0, 100, value=35, step=5, label="Sharpen")
                structure = gr.Slider(0, 100, value=20, step=5, label="Structure / micro-detail")
                clarity = gr.Slider(0, 100, value=0, step=5, label="Edge clarity")
        with gr.Accordion("3 · Color, light and texture", open=False):
            with gr.Row():
                glow = gr.Slider(0, 100, value=0, step=5, label="Highlight glow")
                saturation = gr.Slider(0, 200, value=100, step=5, label="Saturation")
                contrast = gr.Slider(50, 150, value=100, step=5, label="Contrast")
                brightness = gr.Slider(-40, 40, value=0, step=2, label="Brightness")
                grain = gr.Slider(0, 20, value=0, step=1, label="Film grain")
        max_seconds = gr.Slider(0, 86400, value=0, step=1, label="Input clip duration limit in seconds (0 = no app-level limit; runtime limits still apply)")
        with gr.Accordion("Optional · Connect private GPU session", open=False):
            colab_url = gr.Textbox(label="Private GPU connection", placeholder="Paste temporary connection URL", type="password", info="Start your Colab GPU notebook first, then paste its temporary URL here. The value is masked on this page. Leave blank to use this Space’s own processing.")
            gr.Markdown("**Keep the connection private.** Anyone with the temporary URL may be able to access the Colab app. It changes when Colab restarts. The URL must still be copied from Colab once; this app cannot discover a temporary Colab session automatically without a separate always-on broker service.")
        run_btn = gr.Button("✦ ENHANCE VIDEO", variant="primary", size="lg")
        status = gr.Textbox(label="Processing report / diagnostics", lines=5, interactive=False)
        run_btn.click(
            process_video_router,
            inputs=[video_in, model_choice, scale_choice, preset, denoise, sharpen,
                    structure, clarity, glow, saturation, contrast, brightness, grain, max_seconds, colab_url],
            outputs=[video_out, status], show_progress="full", concurrency_limit=1
        )
        gr.Markdown("### QUNQE PROCESS QUEUE · V7")
        gr.Markdown("Save a background job with persistent settings and real frame progress. If Colab disconnects, reopen the notebook with the same Google Drive folder and press Resume. Background work continues while the tab is closed only while the runtime remains alive. Completed videos are saved to the signed-in account’s Processed Videos folder.")
        with gr.Row():
            q_name = gr.Textbox(label="Process name (optional)", placeholder="e.g. Anime edit final")
            q_submit = gr.Button("Save & run in background · QUNQE", variant="primary")
        with gr.Row():
            q_refresh = gr.Button("Refresh process queue")
            q_resume_choice = gr.Dropdown(label="Paused process · video path and percent left (no ID)", choices=[], interactive=True)
            q_resume = gr.Button("Continue selected process", variant="secondary")
        with gr.Row():
            q_saved_video = gr.Dropdown(label="Latest saved leftover/original video path", choices=[], interactive=True)
            q_resume_saved = gr.Button("Resume from selected saved video", variant="secondary")
        q_jobs = gr.JSON(label="QUNQE Process Queue · progress / status / saved checkpoints")
        q_downloads = gr.Files(label="Processed videos for this account")
        q_msg = gr.Markdown("Jobs and progress are stored in the configured persistent job folder.")

        def _job_view(user=None, selected_resume=None, selected_saved=None):
            _cleanup_expired()
            with JOBS_LOCK: jobs=_load_jobs()
            rows=[]; videos=[]; resume_choices=[]; saved_choices=[]
            for jid,j in sorted(jobs.items(), key=lambda item:item[1].get('updated_at',''), reverse=True):
                if not user or j.get('owner') != user: continue
                pct = max(0.0, min(100.0, float(j.get('percent', 0) or 0)))
                remaining = max(0, round(100 - pct))
                rows.append({"process_name":j.get('name','Video process'),"source_video_path":j.get('input_path'),"status":j.get('status'),"percent_complete":pct,"percent_left":remaining,"stage":j.get('stage',''),"error":j.get('error'),"updated_at":j.get('updated_at')})
                if j.get('status') in ('paused','queued','running','failed'):
                    resume_choices.append((f"{j.get('name','Video process')} · {remaining}% left · {j.get('status')} · {j.get('input_path','')}", j.get('input_path')))
                if j.get('status')=='completed' and j.get('output_path') and Path(j['output_path']).is_file(): videos.append(j['output_path'])
            if user:
                qroot=_account_folder(user)/'QUNQE Processes'
                for label, folder in (("LEFTOVER",qroot/'Leftover Videos'),("ORIGINAL",qroot/'Original Videos')):
                    for path in _video_files(folder):
                        associated = next((j for j in jobs.values() if j.get('owner') == user and str(path) in (j.get('input_path'), j.get('leftover_path'))), None)
                        remaining = max(0, round(100 - float(associated.get('percent', 0) or 0))) if associated else None
                        progress_label = f" · {remaining}% left" if label == 'LEFTOVER' and remaining is not None else ''
                        saved_choices.append((f"{label}{progress_label} · {path.name} · {path}", str(path)))
            resume_values=[v for _,v in resume_choices]
            saved_values=[v for _,v in saved_choices]
            resume_value=selected_resume if selected_resume in resume_values else None
            saved_value=selected_saved if selected_saved in saved_values else None
            return rows, videos, gr.update(choices=resume_choices,value=resume_value), gr.update(choices=saved_choices,value=saved_value)

        def _submit_job(video, model, scale, preset_name, dn, sh, st, cl, gl, sat, con, bri, grn, seconds, name, user, selected_resume=None, selected_saved=None):
            if not user: raise gr.Error('Log in to an app account before saving a QUNQE process.')
            if not video: raise gr.Error('Upload a video before saving a process.')
            source=video if isinstance(video,str) else (video.get('path') if isinstance(video,dict) else None)
            if not source or not Path(source).is_file(): raise gr.Error('Could not access the uploaded source video.')
            process_name=_safe_stem(name or Path(source).stem)
            account_root=_account_folder(user); qroot=account_root/'QUNQE Processes'
            original_dir=qroot/'Original Videos'; leftover_dir=qroot/'Leftover Videos'
            original_dir.mkdir(parents=True,exist_ok=True); leftover_dir.mkdir(parents=True,exist_ok=True)
            suffix=Path(source).suffix or '.mp4'; jid=uuid.uuid4().hex
            orig_dest=original_dir/f'{process_name}{suffix}'
            if orig_dest.exists(): orig_dest=original_dir/f'{process_name}_{int(time.time())}{suffix}'
            local_copy=_copy_into_app_storage(user,source,'QUNQE Processes/Original Videos',orig_dest.name)
            if local_copy is None: raise gr.Error('Account storage limit reached. Free space or use a smaller source video.')
            orig_dest=Path(local_copy)
            leftover_dest=leftover_dir/f'{process_name}__LEFTOVER{suffix}'
            if leftover_dest.exists(): leftover_dest=leftover_dir/f'{process_name}_{int(time.time())}__LEFTOVER{suffix}'
            if _dir_bytes(account_root)+orig_dest.stat().st_size > APP_STORAGE_LIMIT:
                orig_dest.unlink(missing_ok=True); raise gr.Error('Not enough account storage for a recovery copy. Free space or use a smaller video.')
            shutil.copy2(orig_dest,leftover_dest)
            (qroot/'Resume Checkpoints'/jid).mkdir(parents=True,exist_ok=True)
            params={"model_name":model,"scale_name":scale,"preset":preset_name,"denoise":dn,"sharpen":sh,"structure":st,"clarity":cl,"glow":gl,"saturation":sat,"contrast":con,"brightness":bri,"grain":grn,"max_seconds":seconds}
            job={"id":jid,"name":process_name,"status":"queued","percent":0.0,"stage":"Saved to QUNQE queue","created_at":_now(),"updated_at":_now(),"input_path":str(orig_dest),"leftover_path":str(leftover_dest),"params":params,"output_path":None,"owner":user}
            with JOBS_LOCK:
                jobs=_load_jobs(); jobs[jid]=job; _save_jobs(jobs)
            JOB_QUEUE.put(jid)
            rows,videos,choices,saved=_job_view(user, selected_resume, selected_saved)
            return rows,videos,choices,saved,f"Saved **{process_name}**. Original and recovery video paths are stable; progress checkpoints are saved separately."

        def _refresh_jobs(user=None, selected_resume=None, selected_saved=None):
            rows,videos,choices,saved=_job_view(user, selected_resume, selected_saved)
            return rows,videos,choices,saved,'Queue refreshed. Selected saved paths are kept when they still exist.'

        def _resume_job(source_path, user, selected_resume=None, selected_saved=None):
            if not user: raise gr.Error('Log in to the same app account to resume a process.')
            source_path=str(source_path or '').strip()
            if not source_path: raise gr.Error('Select a saved video path first.')
            with JOBS_LOCK:
                jobs=_load_jobs()
                matches=[(jid,j) for jid,j in jobs.items() if j.get('owner')==user and source_path in (j.get('input_path'),j.get('leftover_path'))]
                if not matches:
                    source_stem=Path(source_path).stem.split('__LEFT_')[0].split('__LEFTOVER')[0]
                    matches=[(jid,j) for jid,j in jobs.items() if j.get('owner')==user and Path(j.get('input_path','')).stem.split('__LEFT_')[0].split('__LEFTOVER')[0]==source_stem]
                if not matches:
                    raise gr.Error('No saved QUNQE process matches this path. Refresh the queue, or submit the video as a new process.')
                jid,job=matches[0]
                if job.get('status')=='completed': raise gr.Error('This process is already complete. Download it from Processed videos.')
                if not Path(job.get('input_path','')).is_file():
                    # If the original was moved/deleted but the selected recovery copy still exists,
                    # use that identical saved source for the remaining checkpoint work.
                    if Path(source_path).is_file() and source_path in (job.get('leftover_path'), job.get('input_path')):
                        job['input_path']=source_path
                    else:
                        raise gr.Error('Neither the original source nor the selected recovery video exists at the saved path.')
                job['status']='queued'
                job['stage']=f"Resume requested from saved frame checkpoints ({max(0, round(100-float(job.get('percent',0) or 0)))}% left)"
                job['updated_at']=_now(); _save_jobs(jobs)
            JOB_QUEUE.put(jid)
            rows,videos,choices,saved=_job_view(user, selected_resume, selected_saved)
            return rows,videos,choices,saved,f"**{job.get('name','Process')}** queued. It will resume from existing frame checkpoints; check the queue status for any error."

        # Account session is supplied to QUNQE actions; authentication is app-local, not Google sign-in.
        q_submit.click(_submit_job, inputs=[video_in,model_choice,scale_choice,preset,denoise,sharpen,structure,clarity,glow,saturation,contrast,brightness,grain,max_seconds,q_name,account_session,q_resume_choice,q_saved_video], outputs=[q_jobs,q_downloads,q_resume_choice,q_saved_video,q_msg])
        q_refresh.click(_refresh_jobs, inputs=[account_session,q_resume_choice,q_saved_video], outputs=[q_jobs,q_downloads,q_resume_choice,q_saved_video,q_msg])
        q_resume.click(_resume_job, inputs=[q_resume_choice,account_session,q_resume_choice,q_saved_video], outputs=[q_jobs,q_downloads,q_resume_choice,q_saved_video,q_msg])
        q_resume_saved.click(_resume_job, inputs=[q_saved_video,account_session,q_resume_choice,q_saved_video], outputs=[q_jobs,q_downloads,q_resume_choice,q_saved_video,q_msg])
        q_timer = gr.Timer(value=5.0, active=True)
        q_timer.tick(_refresh_jobs, inputs=[account_session,q_resume_choice,q_saved_video], outputs=[q_jobs,q_downloads,q_resume_choice,q_saved_video,q_msg])

        gr.Markdown("---")
        gr.Markdown("## SHIVAM.MS ACCOUNT STORAGE")
        gr.Markdown("Each account has separate Original Videos, Resume Checkpoints, and Processed Videos folders. When the configured storage folder is on Google Drive, these folders and account records persist across runtime resets. Resume uses saved frame checkpoints and lets you choose a process from the list—no process-ID typing is needed.")
        with gr.Row():
            storage_refresh = gr.Button("Refresh app storage")
            storage_status = gr.JSON(label="App storage usage")
        storage_refresh.click(_storage_view, inputs=[account_session], outputs=[storage_status])

        gr.Markdown("""### Included finishing modes
Clean Natural, Crisp Detail, Strong Sharpen, Soft Denoise, Anime Crisp, Texture Pop, Cinematic Contrast, Bright Clean, Glow Highlights, Film Grain, Warm Color, Cool Color, Vivid Color, Matte Film, Edge Clarity, Low Noise.

**Model truth:** real pretrained model weights are downloaded and cached on demand. Only the selected model is loaded into GPU memory when enhancement starts; loading every network into VRAM at once would waste memory and can crash Colab. Finishing modes are image-processing presets, not separate trained AI models. The app checks the runtime at startup and loads the selected model on demand.""")

    def update_boot():
        elapsed = max(0, int(time.time() - BOOT_STARTED))
        left = max(0, BOOT_MIN_SECONDS - elapsed)
        with BOOT_LOCK:
            ready = bool(BOOT["ready"])
            done = int(BOOT.get("done", 0))
            total = max(1, int(BOOT.get("total", 1)))
            status_text = str(BOOT.get("status", "Preparing…"))
            detail = str(BOOT.get("detail", ""))
            error = str(BOOT.get("error", ""))
            frac = float(BOOT.get("fraction", done / total))
        # Reserve last 10% for the mandatory minimum-time gate; actual setup may take longer.
        display_frac = min(0.96, max(0.02, frac * 0.90 + min(elapsed / BOOT_MIN_SECONDS, 1.0) * 0.10))
        mm, ss = divmod(elapsed, 60)
        lm, ls = divmod(left, 60)
        html = f"""<div id='boot-card'><div id='boot-logo'>SM</div><div id='boot-title'>SHIVAM.MS <span style='color:#72e5e0'>AI ENHANCE</span></div><div id='boot-sub'>Preparing your enhancement workspace<br>Checking GPU • preparing video pipeline • loading selected model on demand</div><div id='boot-pulse'></div><div id='boot-meter'><div id='boot-fill' style='width:{display_frac*100:.1f}%'></div></div><div class='boot-stat'>Elapsed: <b>{mm}:{ss:02d}</b></div><div class='boot-stat'>Minimum screen time left: <b>{lm}:{ls:02d}</b></div><div style='margin-top:16px;color:#d8d8ed'>{status_text}</div><div class='boot-note'>{detail}</div></div>"""
        if error:
            html = html.replace("Preparing your enhancement workspace", "Setup needs attention")
        done_gate = ready and left == 0
        if done_gate:
            return gr.update(value=html), gr.update(value=f"**Status:** {status_text}\n\n{detail}" + (f"\n\n**Error:** `{error}`" if error else "")), gr.update(visible=False), gr.update(visible=True), gr.update(active=False)
        return gr.update(value=html), gr.update(value=f"**Status:** {status_text}**\n\n{detail}\n\nRuntime checks complete. Model weights download when you press Enhance. Minimum loading screen: 90 seconds.".replace("**\n", "\n")), gr.update(visible=True), gr.update(visible=False), gr.update(active=True)

    boot_timer.tick(update_boot, inputs=[], outputs=[boot_html, boot_status, boot_screen, main_menu, boot_timer])

if __name__ == "__main__":
    # Set SHIVAM_SHARE=1 for Colab's public temporary Gradio link; on Spaces, leave false.
    demo.queue(max_size=2, default_concurrency_limit=1).launch(
        share=os.getenv("SHIVAM_SHARE", "0") == "1",
        show_error=True,
    )
