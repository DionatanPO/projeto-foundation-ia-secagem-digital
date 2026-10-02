"""Cria ou atualiza um usuário pré-definido (login por e-mail + senha).

Uso:
    python manage.py ensure_user --email usuario@exemplo.com --password "senha-forte"
    python manage.py ensure_user --email admin@exemplo.com --password "senha-forte" --superuser
"""

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Cria ou atualiza um usuário (username = e-mail) com senha válida.'

    def add_arguments(self, parser):
        parser.add_argument('--email', required=True, help='E-mail do usuário (vira username e email).')
        parser.add_argument('--password', required=True, help='Senha inicial.')
        parser.add_argument('--superuser', action='store_true', help='Torna o usuário superuser/staff.')

    def handle(self, *args, **options):
        email = (options['email'] or '').strip().lower()
        password = options['password'] or ''
        if not email or '@' not in email:
            raise CommandError('Informe um --email válido.')
        if len(password) < 8:
            raise CommandError('A senha deve ter ao menos 8 caracteres.')

        user, created = User.objects.get_or_create(
            username=email,
            defaults={'email': email},
        )
        user.email = email
        user.set_password(password)
        if options['superuser']:
            user.is_staff = True
            user.is_superuser = True
        user.save()
        self.stdout.write(self.style.SUCCESS(
            f"{'Criado' if created else 'Atualizado'}: {email}"
            f"{' (superuser)' if user.is_superuser else ''}"
        ))
