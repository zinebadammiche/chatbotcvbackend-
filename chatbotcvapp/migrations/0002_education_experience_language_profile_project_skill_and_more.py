# Generated by Django 4.2 on 2025-01-22 10:40

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('chatbotcvapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Education',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('degree', models.CharField(max_length=255)),
                ('institution', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='Experience',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('company', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='Language',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('language_name', models.CharField(max_length=100, unique=True)),
            ],
        ),
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=254, null=True)),
                ('phone', models.CharField(max_length=20, null=True)),
                ('job_title', models.CharField(max_length=200, null=True)),
                ('linkedin', models.URLField(null=True)),
                ('github', models.URLField(null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Project',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='Skill',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('skill_name', models.CharField(max_length=100)),
                ('category', models.CharField(max_length=100)),
            ],
            options={
                'unique_together': {('skill_name', 'category')},
            },
        ),
        migrations.AddField(
            model_name='cv',
            name='profile',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.profile'),
        ),
        migrations.CreateModel(
            name='ProfileSkill',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.profile')),
                ('skill', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.skill')),
            ],
            options={
                'unique_together': {('profile', 'skill')},
            },
        ),
        migrations.CreateModel(
            name='ProfileProject',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField(null=True)),
                ('profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.profile')),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.project')),
            ],
            options={
                'unique_together': {('profile', 'project')},
            },
        ),
        migrations.CreateModel(
            name='ProfileLanguage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('level', models.CharField(choices=[('A1', 'Débutant'), ('A2', 'Élémentaire'), ('B1', 'Intermédiaire'), ('B2', 'Intermédiaire supérieur'), ('C1', 'Avancé'), ('C2', 'Maîtrise'), ('LM', 'Langue maternelle')], max_length=50)),
                ('language', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.language')),
                ('profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.profile')),
            ],
            options={
                'unique_together': {('profile', 'language')},
            },
        ),
        migrations.CreateModel(
            name='ProfileExperience',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_date', models.DateField()),
                ('end_date', models.DateField(null=True)),
                ('responsibilities', models.TextField()),
                ('experience', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.experience')),
                ('profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.profile')),
            ],
            options={
                'unique_together': {('profile', 'experience', 'start_date')},
            },
        ),
        migrations.CreateModel(
            name='ProfileEducation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_date', models.DateField()),
                ('end_date', models.DateField(null=True)),
                ('education', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.education')),
                ('profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='chatbotcvapp.profile')),
            ],
            options={
                'unique_together': {('profile', 'education')},
            },
        ),
    ]
