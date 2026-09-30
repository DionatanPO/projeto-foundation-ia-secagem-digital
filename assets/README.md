# Assets — AgroMind AI 🌾🧠

Pasta fonte das artes oficiais do projeto. Arquivos oficiais (não apagar):
- `logos/agromind.png` — **logo oficial** (marca AM + fundo transparente, 1288x1221)
- `images/capa.png` — **capa oficial** (logo + "AgroMind AI" sobre fundo azul-marinho, ideal p/ og:image / hero)
- `icons/` + `favicons/` — derivados gerados da logo oficial via PIL (resize com padding quadrado p/ não distorcer)

## Estrutura atual

```
assets/
├── icons/        # icon-192.png, icon-512.png, apple-touch-icon.png (fundo branco)
├── logos/        # agromind.png (OFICIAL)
├── favicons/     # favicon.ico (16/32/48), favicon-16x16.png, favicon-32x32.png
├── images/       # capa.png (OFICIAL), empty-state.svg (ilustração auxiliar)
├── site.webmanifest
└── README.md (este arquivo)
```

Destino usado pelo Django (servido via `{% static %}`) — espelho sincronizado:

```
model_ui/static/model_ui/img/
├── agromind.png / logo.png (cópias da logo oficial)
├── capa.png
├── icon-192.png / icon-512.png / apple-touch-icon.png
├── favicon.ico / favicon-16x16.png / favicon-32x32.png
└── site.webmanifest
```

## Como atualizar a logo no futuro

1. Substitua `assets/logos/agromind.png` pelo novo arquivo (ideal: PNG quadrado, fundo transparente).
2. Regenere os derivados (evita distorção — centraliza em canvas quadrado):
   ```powershell
   venv\Scripts\python.exe -c "
   from PIL import Image
   src = Image.open('assets/logos/agromind.png')
   side = max(src.size)
   sq = Image.new('RGBA', (side, side), (0,0,0,0))
   sq.paste(src, ((side-src.width)//2, (side-src.height)//2), src)
   sq.resize((192,192), Image.LANCZOS).save('assets/icons/icon-192.png')
   sq.resize((512,512), Image.LANCZOS).save('assets/icons/icon-512.png')
   w = sq.resize((180,180), Image.LANCZOS); bg = Image.new('RGB',(180,180),(255,255,255)); bg.paste(w,(0,0),w); bg.save('assets/icons/apple-touch-icon.png')
   sq.resize((16,16), Image.LANCZOS).save('assets/favicons/favicon-16x16.png')
   sq.resize((32,32), Image.LANCZOS).save('assets/favicons/favicon-32x32.png')
   sq.resize((64,64), Image.LANCZOS).save('assets/favicons/favicon.ico', sizes=[(16,16),(32,32),(48,48)])
   "
   ```
3. Copie tudo para o static:
   ```powershell
   Copy-Item assets\logos\agromind.png model_ui\static\model_ui\img\agromind.png -Force
   Copy-Item assets\logos\agromind.png model_ui\static\model_ui\img\logo.png -Force
   Copy-Item assets\images\capa.png model_ui\static\model_ui\img\capa.png -Force
   Copy-Item assets\icons\*.png model_ui\static\model_ui\img\ -Force
   Copy-Item assets\favicons\favicon* model_ui\static\model_ui\img\ -Force
   Copy-Item assets\site.webmanifest model_ui\static\model_ui\img\site.webmanifest -Force
   ```

## Uso nos templates Django

```html
{% load static %}
<link rel="icon" href="{% static 'model_ui/img/favicon.ico' %}" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="{% static 'model_ui/img/favicon-32x32.png' %}">
<link rel="apple-touch-icon" href="{% static 'model_ui/img/apple-touch-icon.png' %}">
<link rel="manifest" href="{% static 'model_ui/img/site.webmanifest' %}">
<img src="{% static 'model_ui/img/agromind.png' %}" alt="AgroMind AI" width="36" height="36">
<meta property="og:image" content="{% static 'model_ui/img/capa.png' %}">
```

Onde a logo aparece hoje:
- `model_ui/templates/model_ui/index.html` — favicon/manifest/og + `<img>` na sidebar
- `model_ui/templates/model_ui/login.html` — favicon + `<img>` no card de login
