# SHIVAM.MS AI ENHANCE — private ZIP uploader / auto-unlocker / launcher
# Paste this entire cell into Google Colab and run it.
# Supports filenames beginning SHIVAM_MS_AI_ENHANCE_V followed by any version number.
# Requires a password-protected ZIP made with the matching project password.

import os, sys, re, zipfile, shutil, subprocess, contextlib, io, runpy
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Quiet mode: no public Gradio share URL, no analytics, and no ordinary pip logs.
os.environ['GRADIO_SHARE'] = 'False'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['SHIVAM_RUNTIME'] = 'colab'
os.environ['SHIVAM_SHARE'] = '0'

ROOT = Path('/content/SHIVAM_MS_AI_ENHANCE_PRIVATE')
ZIP_PATH = Path('/content/SHIVAM_MS_AI_ENHANCE_UPLOAD.zip')
MAX_ZIP_BYTES = 2 * 1024**3
# Encoded locally to avoid printing the password in the UI. This is obfuscation,
# not strong secrecy: anyone who can edit/read this notebook can recover it.
_PASSWORD = bytes.fromhex('53484956414d2e4d53313131313131')

# Cool upload panel

display(HTML('''
<div style="max-width:780px;margin:14px auto;padding:26px;border-radius:22px;
background:linear-gradient(135deg,#0b1020,#172554 58%,#164e63);color:#f8fafc;
font-family:Arial;border:1px solid #334155;box-shadow:0 12px 34px #0003">
 <div style="font-size:11px;letter-spacing:3px;color:#7dd3fc">SHIVAM.MS • PRIVATE WORKSPACE</div>
 <h1 style="margin:10px 0 6px;font-size:29px">AI ENHANCE <span style="color:#67e8f9">V7</span></h1>
 <div style="color:#cbd5e1">Secure ZIP check · Auto-unlock · Confirm · Launch</div>
 <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:18px">
  <span style="background:#ffffff12;padding:8px 11px;border-radius:10px">📦 ZIP uploader</span>
  <span style="background:#ffffff12;padding:8px 11px;border-radius:10px">🔐 Auto-unlock</span>
  <span style="background:#ffffff12;padding:8px 11px;border-radius:10px">🔗 Public sharing off</span>
 </div>
</div>'''))

upload = widgets.FileUpload(accept='.zip', multiple=False, description='📦 Choose V-version ZIP',
                            layout=widgets.Layout(width='100%'))
confirm = widgets.Checkbox(value=False, description='I confirm this is my SHIVAM_MS_AI_ENHANCE_V… ZIP', indent=False)
launch = widgets.Button(description='🚀  UNLOCK & LAUNCH', button_style='success', disabled=True,
                        layout=widgets.Layout(width='100%', height='46px'))
status = widgets.HTML(value='<div style="padding:12px;color:#94a3b8">Choose a password-protected V-version ZIP.</div>')

def get_upload():
    val = upload.value
    if not val:
        return None, None
    # ipywidgets 7 returns a tuple/list; ipywidgets 8 returns a mapping.
    if isinstance(val, dict):
        name, item = next(iter(val.items()))
        name = item.get('metadata', {}).get('name', name)
        return name, bytes(item.get('content', b''))
    item = val[0]
    return item.get('name', ''), bytes(item.get('content', b''))

def check_upload(*_):
    name, data = get_upload()
    valid_name = bool(name and re.match(r'^SHIVAM_MS_AI_ENHANCE_V\d+.*\.zip$', name, re.I))
    valid_size = bool(data and len(data) <= MAX_ZIP_BYTES)
    launch.disabled = not (valid_name and valid_size and confirm.value)
    if name and not valid_name:
        status.value = '<div style="padding:12px;color:#fca5a5">Wrong file: filename must start with SHIVAM_MS_AI_ENHANCE_V followed by a version number.</div>'
    elif valid_name and valid_size:
        status.value = f'<div style="padding:12px;color:#86efac">✓ Selected: {name} · {len(data)/1024:.1f} KB. Confirm to continue.</div>'
    elif data and not valid_size:
        status.value = '<div style="padding:12px;color:#fca5a5">Wrong file: ZIP is empty or larger than the 2 GB limit.</div>'

upload.observe(check_upload, names='value')
confirm.observe(check_upload, names='value')

def launch_app(_):
    launch.disabled = True
    upload.disabled = True
    confirm.disabled = True
    status.value = '<div style="padding:12px;color:#7dd3fc">Checking ZIP lock and validating files…</div>'
    try:
        name, data = get_upload()
        if not name or not re.match(r'^SHIVAM_MS_AI_ENHANCE_V\d+.*\.zip$', name, re.I):
            raise ValueError('WRONG_FILE')
        ZIP_PATH.write_bytes(data)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            infos = zf.infolist()
            if not infos or any(not (i.flag_bits & 1) for i in infos):
                raise ValueError('NOT_LOCKED')
            if len(infos) > 10000:
                raise ValueError('WRONG_FILE')
            if any(Path(i.filename).is_absolute() or '..' in Path(i.filename).parts for i in infos):
                raise ValueError('WRONG_FILE')
            # Wrong password and corrupt/unsupported encryption are handled below.
            try:
                zf.setpassword(_PASSWORD)
                bad = zf.testzip()
                if bad is not None:
                    raise ValueError('WRONG_PASSWORD')
                if ROOT.exists():
                    shutil.rmtree(ROOT)
                ROOT.mkdir(parents=True, exist_ok=True)
                zf.extractall(ROOT, pwd=_PASSWORD)
            except RuntimeError:
                raise ValueError('WRONG_PASSWORD')
            except zipfile.BadZipFile:
                raise ValueError('WRONG_FILE')

        app_files = list(ROOT.rglob('app.py'))
        if not app_files:
            raise ValueError('NO_APP')
        app_file = next((p for p in app_files if (p.parent/'models.py').exists()
                         and (p.parent/'effects.py').exists()), app_files[0])
        app_dir = app_file.parent
        os.chdir(app_dir)
        sys.path.insert(0, str(app_dir)) if str(app_dir) not in sys.path else None

        # Mount Drive in Colab so account records, originals, checkpoints, leftovers,
        # processed videos and model weights survive runtime resets.
        storage_root = Path('/content/SHIVAM_MS_AI_ENHANCE_STORAGE')
        try:
            from google.colab import drive
            if not Path('/content/drive/MyDrive').exists():
                drive.mount('/content/drive', force_remount=False)
            if Path('/content/drive/MyDrive').exists():
                storage_root = Path('/content/drive/MyDrive/SHIVAM_MS_AI_ENHANCE')
        except Exception as drive_error:
            print('Google Drive could not be mounted; using temporary Colab storage. Details:', drive_error)
        storage_root.mkdir(parents=True, exist_ok=True)
        storage_root.mkdir(parents=True, exist_ok=True)
        os.environ['SHIVAM_JOB_DIR'] = str(storage_root/'jobs')
        os.environ['SHIVAM_APP_STORAGE'] = str(storage_root/'accounts')
        os.environ['SHIVAM_MODEL_DIR'] = str(storage_root/'models')
        os.environ['SHIVAM_OUTPUT_DIR'] = str(storage_root/'processed')
        os.environ['SHIVAM_SHARE'] = '0'

        req = app_dir/'requirements.txt'
        if req.exists():
            p = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(req)])
            if p.returncode:
                raise RuntimeError('Dependency installation failed. Review the pip output above.')

        import gradio as gr
        original_launch = gr.Blocks.launch
        def private_launch(self, *args, **kwargs):
            kwargs['share'] = False
            kwargs['inbrowser'] = False
            kwargs['quiet'] = True
            kwargs['show_error'] = True
            # Inline rendering is supported only by some Gradio/Colab versions.
            kwargs.setdefault('inline', True)
            return original_launch(self, *args, **kwargs)
        gr.Blocks.launch = private_launch
        clear_output(wait=True)
        display(HTML('<div style="max-width:780px;margin:16px auto;padding:20px;background:#0b1020;color:#f8fafc;border-radius:16px;font-family:Arial"><div style="color:#86efac;font-size:12px;letter-spacing:2px">ZIP VERIFIED</div><h2>SHIVAM.MS AI ENHANCE</h2><p>Starting app in this notebook. Public sharing is disabled.</p></div>'))
        try:
            runpy.run_path(str(app_file), run_name='__main__')
        finally:
            gr.Blocks.launch = original_launch

    except ValueError as e:
        reason = str(e)
        messages = {
            'NOT_LOCKED': 'WRONG FILE: this ZIP is not password-protected.',
            'WRONG_PASSWORD': 'INCORRECT PASSWORD or unsupported ZIP encryption.',
            'WRONG_FILE': 'WRONG FILE: invalid ZIP, filename, or unsafe contents.',
            'NO_APP': 'WRONG FILE: app.py was not found in this ZIP.',
            'DEPENDENCIES': 'V7 could not start because dependency installation failed.'
        }
        status.value = '<div style="padding:14px;color:#fca5a5">' + messages.get(reason, 'V7 could not start. Check the ZIP and try again.') + '</div>'
        upload.disabled = False
        confirm.disabled = False
        launch.disabled = False
    except Exception as exc:
        status.value = '<div style="padding:14px;color:#fca5a5">V7 could not start: ' + str(exc).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;') + '</div>'
        print('SHIVAM.MS launcher error:', repr(exc))
        upload.disabled = False
        confirm.disabled = False
        launch.disabled = False

launch.on_click(launch_app)
display(widgets.VBox([upload, confirm, launch, status], layout=widgets.Layout(max_width='780px', margin='12px auto', gap='12px')))
