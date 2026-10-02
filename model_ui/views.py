from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.views.decorators.csrf import ensure_csrf_cookie


def _get_username_for_email(email):
    try:
        return User.objects.get(email__iexact=email).username
    except (User.DoesNotExist, User.MultipleObjectsReturned):
        return email


def login_view(request):
    if request.user.is_authenticated:
        return redirect('chat-interface')

    error = None
    if request.method == 'POST':
        email = (request.POST.get('email') or '').strip()
        password = request.POST.get('password') or ''
        if not email or not password:
            error = 'Informe e-mail e senha.'
        else:
            user = authenticate(
                request,
                username=_get_username_for_email(email),
                password=password,
            )
            if user is not None:
                login(request, user)
                request.session['just_logged_in'] = True
                return redirect('chat-interface')
            error = 'E-mail ou senha inválidos. Tente novamente.'

    return render(request, 'model_ui/login.html', {'error': error})


def logout_view(request):
    logout(request)
    return redirect('login')


@ensure_csrf_cookie
def chat_interface(request):
    if not request.user.is_authenticated:
        return redirect('login')
    show_welcome = bool(request.session.pop('just_logged_in', False))
    return render(request, 'model_ui/index.html', {'show_welcome': show_welcome})


def landing_view(request):
    return render(request, 'model_ui/landing.html')
