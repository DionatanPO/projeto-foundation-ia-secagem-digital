from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def _drop_orphan_conversations(apps, schema_editor):
    Conversation = apps.get_model('api', 'Conversation')
    # Conversas antigas (globais, sem dono) não têm como ser atribuídas com
    # segurança a um usuário — e o banco atual está vazio. Remove órfãs.
    Conversation.objects.filter(user__isnull=True).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_conversation_chatmessage'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name='conversation',
            name='user',
            field=models.ForeignKey(
                null=True,
                blank=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name='conversations',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.RunPython(_drop_orphan_conversations, migrations.RunPython.noop),
        migrations.AlterField(
            model_name='conversation',
            name='user',
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name='conversations',
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
