from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import User
from django.utils import timezone

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # L'utilisateur qui a initié la conversation
    message = models.TextField()  # Le message de l'utilisateur
    response = models.TextField()  # La réponse du chatbot
    created_at = models.DateTimeField(auto_now_add=True)  # Date et heure de la conversation

class CV(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # L'utilisateur associé au CV
    profile = models.ForeignKey('Profile', on_delete=models.CASCADE, null=True, blank=True)  # Le profil associé au CV
    path = models.CharField(max_length=255)  # Chemin du fichier CV
    upload_date = models.DateTimeField(auto_now_add=True)  # Date de téléversement du CV

    def __str__(self):
        return f"CV de {self.user.username}"

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # L'utilisateur associé au profil
    name = models.CharField(max_length=100)
    email = models.EmailField(null=True)
    phone = models.CharField(max_length=20, null=True)
    job_title = models.CharField(max_length=200, null=True)
    linkedin = models.URLField(null=True)
    github = models.URLField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Skill(models.Model):
    skill_name = models.CharField(max_length=100)
    category = models.CharField(max_length=100)  # ex: "Langages de programmation", "Frameworks"

    class Meta:
        unique_together = ('skill_name', 'category')

    def __str__(self):
        return f"{self.category}: {self.skill_name}"

class ProfileSkill(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    skill = models.ForeignKey(Skill, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('profile', 'skill')

class Experience(models.Model):
    title = models.CharField(max_length=255)
    company = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.title} chez {self.company}"

class ProfileExperience(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    experience = models.ForeignKey(Experience, on_delete=models.CASCADE)
    start_date = models.DateField()
    end_date = models.DateField(null=True)
    responsibilities = models.TextField()

    class Meta:
        unique_together = ('profile', 'experience', 'start_date')

class Education(models.Model):
    degree = models.CharField(max_length=255)
    institution = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.degree} - {self.institution}"

class ProfileEducation(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    education = models.ForeignKey(Education, on_delete=models.CASCADE)
    start_date = models.DateField()
    end_date = models.DateField(null=True)

    class Meta:
        unique_together = ('profile', 'education')

class Language(models.Model):
    language_name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.language_name

class ProfileLanguage(models.Model):
    LEVEL_CHOICES = [
        ('A1', 'Débutant'),
        ('A2', 'Élémentaire'),
        ('B1', 'Intermédiaire'),
        ('B2', 'Intermédiaire supérieur'),
        ('C1', 'Avancé'),
        ('C2', 'Maîtrise'),
        ('LM', 'Langue maternelle'),
    ]

    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    language = models.ForeignKey(Language, on_delete=models.CASCADE)
    level = models.CharField(max_length=50, choices=LEVEL_CHOICES)

    class Meta:
        unique_together = ('profile', 'language')

class Project(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class ProfileProject(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    year = models.IntegerField(null=True)

    class Meta:
        unique_together = ('profile', 'project')