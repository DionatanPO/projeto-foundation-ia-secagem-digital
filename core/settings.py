from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-8tq=*#r2brek!81j(idb1w%!u4b&oglhxpw4093eqctuqyn@y^')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['127.0.0.1', 'localhost', 'ai.secagemdigital.com']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
    'model_ui',
    'live',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = False
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'static'
CORS_ALLOW_ALL_ORIGINS = True
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.SessionAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.AllowAny',
    ),
}

# Model Configuration (LMM)
MODEL_PATH = os.getenv('MODEL_PATH', './models/seu_modelo_aqui.gguf')
MMPROJ_PATH = os.getenv('MMPROJ_PATH', None)
N_THREADS = int(os.getenv('N_THREADS', 4))
N_CTX = int(os.getenv('N_CTX', 16384))
USE_FLASH_ATTN = os.getenv('USE_FLASH_ATTN', 'True').lower() == 'true'
N_GPU_LAYERS = int(os.getenv('N_GPU_LAYERS', '0'))

# HuggingFace
HF_TOKEN = os.getenv('HF_TOKEN', None)

# Modo Live (voz Piper) — módulo isolado, não afeta o chat
PIPER_VOICE = os.getenv('PIPER_VOICE', 'pt_BR-faber-medium')
PIPER_NOISE_SCALE = os.getenv('PIPER_NOISE_SCALE', '0.8')
PIPER_NOISE_W = os.getenv('PIPER_NOISE_W', '0.8')

# Modo Live — transcrição local (Faster-Whisper, CPU). Modelo baixa no 1º uso.
LIVE_STT_MODEL = os.getenv('LIVE_STT_MODEL', 'small')  # tiny|base|small|medium
LIVE_STT_DIR = os.getenv('LIVE_STT_DIR', str(BASE_DIR / 'models' / 'stt'))
LIVE_STT_DEVICE = os.getenv('LIVE_STT_DEVICE', 'cpu')
LIVE_STT_COMPUTE = os.getenv('LIVE_STT_COMPUTE', 'int8')
PIPER_MODEL_DIR = os.getenv('PIPER_MODEL_DIR', str(BASE_DIR / 'models' / 'tts'))
PIPER_CACHE_DIR = os.getenv('PIPER_CACHE_DIR', str(BASE_DIR / 'storage' / 'tts_cache'))
LIVE_MAX_CHARS = int(os.getenv('LIVE_MAX_CHARS', '600'))
