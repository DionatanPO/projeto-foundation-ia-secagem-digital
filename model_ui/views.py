import os
from django.shortcuts import render, redirect


def _check_key(key):
    return key == os.getenv('APP_KEY', '')


def login_view(request):
    if request.session.get('authenticated'):
        return redirect('chat-interface')

    error = None
    if request.method == 'POST':
        key = request.POST.get('app_key', '').strip()
        if _check_key(key):
            request.session['authenticated'] = True
            return redirect('chat-interface')
        else:
            error = 'Chave inválida. Tente novamente.'

    return render(request, 'model_ui/login.html', {'error': error})


def chat_interface(request):
    if not request.session.get('authenticated'):
        return redirect('login')
    return render(request, 'model_ui/index.html')
