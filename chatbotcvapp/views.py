
# Importations nécessaires
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.authentication import TokenAuthentication
from rest_framework.views import APIView
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.db import transaction
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import (
    CV, Skill, ProfileSkill, Experience, ProfileExperience,
    Education, ProfileEducation, Profile, Project, ProfileProject, Language, ProfileLanguage
)
import time 
import logging
from .utils.pdf_extractor import PDFCVExtractor
import os
import tempfile
from datetime import datetime
import re
import spacy
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import PyPDF2
import pickle
from langchain import HuggingFaceHub
import string
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from django.contrib.auth.models import User
from .models import Chat, CV, Profile 
import tempfile
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import PyPDF2
import pickle
from langchain import HuggingFaceHub
from rest_framework.response import Response
from .models import Chat 

from rest_framework.decorators import api_view, permission_classes,authentication_classes
 
import logging
from django.utils import timezone
from datetime import timedelta
import os
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Charger le modèle HuggingFace
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={
        "max_length": 1000,  # Longueur maximale de la séquence générée
        "max_new_tokens": 10000,  # Nombre maximal de nouveaux tokens à générer
    },
    huggingfacehub_api_token=huggingface_token # Remplacez par votre clé API
)


# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

# Charger le modèle et le vectoriseur TF-IDF
clf = pickle.load(open('C:\\Users\\adm19\\chatbotcv\\chatbotcvapp\\utils\\clf.pkl', 'rb'))
tfidf = pickle.load(open('C:\\Users\\adm19\\chatbotcv\\chatbotcvapp\\utils\\tfidf.pkl', 'rb'))

# Dictionnaire de correspondance des catégories
category_mapping = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing',
}

# Regex pour identifier les adresses e-mail
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Fonction pour nettoyer le texte
def clear_fun(text):
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', ' ', text)
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    return text

# Fonction pour extraire le texte de plusieurs PDFs
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                texts.append(text)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {pdf_path}: {e}")
    return texts

# Fonction pour extraire le prénom et le nom du texte du CV
def extract_name_from_text(text):
    doc = nlp(text)
    first_name = None
    last_name = None
    for ent in doc.ents:
        if ent.label_ == "PER":  # PER = Personne
            parts = ent.text.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = " ".join(parts[1:])
                break
    return first_name + last_name

# Fonction pour classifier plusieurs CVs
def classify_resumes(resume_texts):
    predicted_categories = []
    for resume_text in resume_texts:
        cleaned_resume = clear_fun(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        predicted_category = category_mapping.get(prediction_id, 'Unknown')
        predicted_categories.append(predicted_category)
    return predicted_categories

# Fonction pour envoyer un e-mail à une catégorie spécifique
def send_email_to_category(category, pdf_paths, email_subject, email_body):
    for pdf_path in pdf_paths:
        resume_text = extract_text_from_pdfs([pdf_path])[0]
        predicted_category = classify_resumes([resume_text])[0]
        if predicted_category == category:
            emails = re.findall(EMAIL_REGEX, resume_text)
            sender_email = "recruitb18@gmail.com"
            sender_password = "vnzz ovqe tkuu srbf"
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(sender_email, sender_password)
                for email in emails:
                    msg = MIMEMultipart()
                    msg['From'] = sender_email
                    msg['To'] = email
                    msg['Subject'] = email_subject
                    msg.attach(MIMEText(email_body, 'plain'))
                    server.sendmail(sender_email, email, msg.as_string())
                    print(f"E-mail envoyé à {email} depuis le fichier {pdf_path}")
                server.quit()
            except Exception as e:
                print(f"Erreur lors de l'envoi des e-mails pour le fichier {pdf_path} : {e}")
        else:
            print(f"Aucun CV trouvé dans la catégorie {category} pour le fichier {pdf_path}.")

# Vue pour l'inscription
@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    username = request.data.get('username')
    password = request.data.get('password')
    email = request.data.get('email')

    if not username or not password or not email:
        return Response({'error': 'Veuillez fournir un nom d\'utilisateur, un mot de passe et un e-mail.'}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username=username).exists():
        return Response({'error': 'Ce nom d\'utilisateur existe déjà.'}, status=status.HTTP_400_BAD_REQUEST)

    user = User.objects.create_user(username=username, password=password, email=email)
    token, _ = Token.objects.get_or_create(user=user)

    return Response({'token': token.key}, status=status.HTTP_201_CREATED)
# Vue pour la connexion
@api_view(['POST'])
@permission_classes([AllowAny])
def user_login(request):
    email = request.data.get('email')
    password = request.data.get('password')

    if not email or not password:
        return Response({'error': 'Veuillez fournir un e-mail et un mot de passe.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = User.objects.get(email=email)
    except User.DoesNotExist:
        return Response({'error': 'Identifiants invalides.'}, status=status.HTTP_400_BAD_REQUEST)

    authenticated_user = authenticate(username=user.username, password=password)

    if authenticated_user:
        token, _ = Token.objects.get_or_create(user=authenticated_user)
        return Response({'token': token.key, 'username': user.username}, status=status.HTTP_200_OK)
    else:
        return Response({'error': 'Identifiants invalides.'}, status=status.HTTP_400_BAD_REQUEST)

# Vue pour la déconnexion
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def user_logout(request):
    logout(request)
    return Response({'message': 'Logged out successfully'}, status=status.HTTP_200_OK)

# Vue pour l'upload et le traitement des CVs
class UploadCVAPIView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        try:
            user = request.user
            cv_file = request.FILES['cv']
            print(cv_file.name)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, cv_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                for chunk in cv_file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name

            if not os.path.exists(temp_path):
                return Response({
                    'status': 'error',
                    'message': f"Le fichier temporaire {temp_path} n'existe pas.",
                }, status=status.HTTP_400_BAD_REQUEST)

            resume_text = extract_text_from_pdfs([temp_path])[0]
            predicted_category = classify_resumes([resume_text])[0]
            new_filename = f"{extract_name_from_text(resume_text)}_{predicted_category}.pdf"

            with open(temp_path, 'wb') as f:
                for chunk in cv_file.chunks():
                    f.write(chunk)
            # Enregistrer le fichier dans le dossier `media`
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            saved_path = fs.save(new_filename, cv_file)

            extractor = PDFCVExtractor()
            cv_data = extractor.extract_from_pdf(temp_path)
            skills_data = extractor.extract_skills_from_pdf(temp_path)
            experiences_data = extractor.extract_experiences_from_pdf(temp_path)
            educations_data = extractor.extract_education_from_pdf(temp_path)
            projects_data = extractor.extract_projects_from_pdf(temp_path)
            languages_data = extractor.extract_languages_from_pdf(temp_path)

            profile, profile_created = Profile.objects.update_or_create(
                user=user,
                defaults={
                    'name': cv_data.get('name', user.profile.name if hasattr(user, 'profile') else 'Inconnu'),
                    'email': cv_data.get('email', user.profile.email if hasattr(user, 'profile') else ''),
                    'phone': cv_data.get('phone', user.profile.phone if hasattr(user, 'profile') else ''),
                    'job_title': cv_data.get('job_title', user.profile.job_title if hasattr(user, 'profile') else ''),
                    'linkedin': cv_data.get('linkedin', user.profile.linkedin if hasattr(user, 'profile') else ''),
                    'github': cv_data.get('github', user.profile.github if hasattr(user, 'profile') else '')
                }
            )

            cv_instance = CV.objects.create(user=user, profile=profile, path=new_filename)

            os.remove(temp_path)
            os.rmdir(temp_dir)

            return Response({
                'status': 'success',
                'message': 'CV traité avec succès',
                'user_id': user.id,
                'profile_id': profile.id,
                'cv_id': cv_instance.id,
                'file_path': saved_path  # Retourner le chemin du fichier enregistré
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                'status': 'error',
                'message': f"Erreur lors du traitement du CV : {str(e)}",
                'details': str(e.__class__.__name__)
            }, status=status.HTTP_400_BAD_REQUEST)
def extract_subject_and_body_with_mistral(message):
    """
    Utilise Mistral pour extraire le sujet et le corps de l'e-mail à partir du message de l'utilisateur.
    
    Args:
        message (str): Le message de l'utilisateur contenant les informations pour l'e-mail.
    
    Returns:
        tuple: Un tuple contenant le sujet et le corps de l'e-mail.
    """
    try:
        # Préparer le prompt pour Mistral
        prompt = (
            "Extrait le sujet et le corps d'un e-mail à partir du message suivant. "
            "Le message peut contenir des instructions comme 'envoyer un mail à la catégorie X' ou d'autres détails. "
            "Retourne uniquement le sujet et le corps de l'e-mail, sans commentaires supplémentaires.\n\n"
            f"Message : {message}\n\n"
            "Sujet :"
        )

        # Appeler le modèle Mistral
        response = llm.invoke(prompt)
        
        # Extraire le sujet et le corps de la réponse
        subject = response.split("\n")[0].replace("Sujet :", "").strip()
        body = "\n".join(response.split("\n")[1:]).strip()
        
        return subject, body
    except Exception as e:
        print(f"Erreur lors de l'appel à Mistral : {e}")
        # Retourner des valeurs par défaut en cas d'erreur
        return "Opportunité de recrutement", "Bonjour,\n\nNous avons une opportunité intéressante pour vous.\n\nCordialement,\nL'équipe de recrutement."
class ChatbotAPIView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def detect_intention(self, message):
        message = message.lower().strip()  # Convertir en minuscules et supprimer les espaces inutiles
        print(f"Message reçu : {message}")

        # Si le message est vide, retourner "extract_data"
        if not message:
            return "extract_data"

        # Détection de l'intention "send_email"
        if re.search(r'envoy(er|ez) (un e-mail|un mail) (à|a) (la catégorie|la category|un candidat)', message):
            return "send_email"

        # Détection de l'intention "classify_cv"
        elif re.search(r'classif(ier|iez)|catégorie|classification|categoriser', message):
            return "classify_cv"

        # Détection de l'intention "extract_data"
        elif re.search(r'extraire|données|extract', message):
            return "extract_data"

        # Si le message ne correspond à aucune intention connue, retourner "ask_question"
        else:
            return "ask_question"

    def post(self, request, *args, **kwargs):
        user = request.user
        message = request.data.get('message', '').strip().lower()
        cv_files = request.FILES.getlist('cv')

        try:
            intention = self.detect_intention(message)
            print(f"Message utilisateur : {message}")
            print(f"Intention détectée : {intention}")

            if intention == "classify_cv":
                if not cv_files:
                    return Response({"status": "error", "message": "Aucun fichier CV fourni."}, status=status.HTTP_400_BAD_REQUEST)

                temp_paths = []
                for cv_file in cv_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        for chunk in cv_file.chunks():
                            temp_file.write(chunk)
                        temp_paths.append(temp_file.name)
                resume_texts = extract_text_from_pdfs(temp_paths)
                predicted_categories = classify_resumes(resume_texts)

                results = []
                for i, predicted_category in enumerate(predicted_categories):
                    results.append({
                        'filename': cv_files[i].name,
                        'category': predicted_category
                    })
 # Créer un prompt pour Mistral
                prompt = (
                    "Vous êtes un assistant spécialisé dans le recrutement et l'analyse de CVs. "
                    "Voici les résultats de la classification des CVs :\n\n"
                )
              # Fonction pour appeler le modèle Mistral avec réessai en cas d'échec
                def call_mistral_with_retry(prompt, max_retries=3, delay=2):
                    for attempt in range(max_retries):
                        try:
                            response = llm.invoke(prompt)
                            return response
                        except Exception as e:
                            logger.warning(f"Tentative {attempt + 1} échouée : {str(e)}")
                            time.sleep(delay)
                    raise Exception("Échec après plusieurs tentatives d'appel à l'API Mistral")

                # Votre code existant
                for result in results:
                    prompt += f"- Le fichier {result['filename']} a été classé dans la catégorie {result['category']}.\n"

                prompt += (
                    "\nReformulez ces résultats en une phrase cohérente et professionnelle, "
                    "en combinant les informations du CV et de la catégorie."
                )
                print(prompt)

                # Appeler le modèle Mistral pour reformuler les résultats
                try:
                    reformulated_response = call_mistral_with_retry(prompt)
                    print(f"Réponse reformulée par Mistral : {reformulated_response}")
                    # Extraire la partie après le deuxième \n\n
                    reformulated_response = reformulated_response.split("\n\n")[-1]
                except Exception as e:
                    reformulated_response = "Erreur lors de la reformulation des résultats."
                    print(f"Erreur lors de l'appel du modèle Mistral : {str(e)}")


                for temp_path in temp_paths:
                    os.remove(temp_path)

                Chat.objects.create(
                    user=user,
                    message=message,
                    response=reformulated_response
                )

                return Response({
                    "status": "success",
                    "results": results,
                    "response": reformulated_response
                }, status=status.HTTP_200_OK)
            elif intention == "extract_data":
                if not cv_files:
                    return Response({"status": "error", "message": "Aucun fichier CV fourni."}, status=status.HTTP_400_BAD_REQUEST)

                try:
                    extracted_data = []
                    cv_summaries = []  # Pour stocker les résumés de chaque CV

                    for cv_file in cv_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            for chunk in cv_file.chunks():
                                temp_file.write(chunk)
                            temp_path = temp_file.name

                        try:
                            # Extraire les données du CV
                            extractor = PDFCVExtractor()
                            cv_data = extractor.extract_from_pdf(temp_path)
                            extracted_data.append(cv_data)

                            # Classifier le CV de manière implicite
                            resume_text = extract_text_from_pdfs([temp_path])[0]
                            predicted_category = classify_resumes([resume_text])[0]

                            # Générer un nom de fichier basé sur le nom du profil et la catégorie
                            name = cv_data.get('name', 'Unknown')
                            new_filename = f"{name.replace(' ', '_')}_{predicted_category}.pdf"

                            # Mettre à jour ou créer le profil
                            profile, profile_created = Profile.objects.update_or_create(
                                user=user,
                                defaults={
                                    'name': cv_data.get('name', user.profile.name if hasattr(user, 'profile') else 'Inconnu'),
                                    'email': cv_data.get('email', user.profile.email if hasattr(user, 'profile') else ''),
                                    'phone': cv_data.get('phone', user.profile.phone if hasattr(user, 'profile') else ''),
                                    'job_title': cv_data.get('job_title', user.profile.job_title if hasattr(user, 'profile') else ''),
                                    'linkedin': cv_data.get('linkedin', user.profile.linkedin if hasattr(user, 'profile') else ''),
                                    'github': cv_data.get('github', user.profile.github if hasattr(user, 'profile') else '')
                                }
                            )
                            skills_data = extractor.extract_skills_from_pdf(temp_path)
                            # Enregistrer le fichier dans le dossier `media`
                            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
                            saved_path = fs.save(new_filename, cv_file)
                            CV.objects.create(user=user, profile=profile, path=new_filename)

                            # Ajouter un résumé pour ce CV
                            cv_summaries.append(
                                f"- {name} : CV enregistré sous le nom de fichier {new_filename}. "
                                f"Catégorie : {predicted_category}. "
                                f"Compétences : {skills_data }."
                            )

                        finally:
                            # Supprimer le fichier temporaire
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                        # Générer un prompt unique pour tous les CV
                    prompt = (
        f"Voici un résumé des CV extraits :\n"
        f"{chr(10).join(cv_summaries)}\n"  # Use chr(10) to represent a newline character
        f"Écris un paragraphe concis (4-5 phrases) résumant les informations principales de tous les CV."
    )
                    # Appel au modèle Mistral
                    mistral_response = llm.invoke(prompt)
                    print(mistral_response)  # Pour déboguer
                    response = mistral_response.split("\n\n")[1] if "\n\n" in mistral_response else mistral_response

                    # Enregistrer la réponse unique dans la base de données
                    Chat.objects.create(
                        user=user,
                        message="Extraction des CV",
                        response=response
                    )

                    return Response({
                        'status': 'success',
                        'message': 'Données extraites avec succès et enregistrées dans la base de données.',
                        'data': extracted_data,
                        'response': response  # Renvoyer la réponse unique
                    }, status=status.HTTP_200_OK)

                except Exception as e:
                    return Response({
                        'status': 'error',
                        'message': f"Erreur lors de l'extraction des données : {str(e)}",
                        'details': str(e.__class__.__name__)
                    }, status=status.HTTP_400_BAD_REQUEST)
            elif intention == "send_email":
                # Détecter la catégorie spécifiée dans le message
                category_match = re.search(r'catégorie (.+)', message)
                if not category_match:
                    return Response({"status": "error", "message": "Veuillez spécifier une catégorie."}, status=status.HTTP_400_BAD_REQUEST)

                category = category_match.group(1).strip()

                if not cv_files:
                    return Response({"status": "error", "message": "Aucun fichier CV fourni."}, status=status.HTTP_400_BAD_REQUEST)

                temp_paths = []
                for cv_file in cv_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        for chunk in cv_file.chunks():
                            temp_file.write(chunk)
                        temp_paths.append(temp_file.name)

                # Extraire le texte des CV
                resume_texts = extract_text_from_pdfs(temp_paths)
                predicted_categories = classify_resumes(resume_texts)

                subject, body = extract_subject_and_body_with_mistral(message)
               
                

                emails_sent = 0
                sent_emails_list = []  # Liste pour stocker les e-mails envoyés

                # Envoyer les e-mails aux candidats de la catégorie spécifiée
                for i, predicted_category in enumerate(predicted_categories):
                    if predicted_category.lower() == category.lower():
                        emails = re.findall(EMAIL_REGEX, resume_texts[i])
                        if not emails:
                            print(f"Aucune adresse e-mail trouvée dans le fichier {temp_paths[i]}")
                            continue
                                # Extraire les données du CV
                        extractor = PDFCVExtractor()
                        cv_data = extractor.extract_from_pdf(temp_paths[0])  # Utilisez le premier CV pour extraire le nom
                        name = cv_data.get('name' )  # Fonction pour extraire le nom du CV

                        # Définir email_body avant de l'utiliser
                        email_body = body.replace("[Nom du candidat]", name)
                        email_subject = subject
                        print(email_subject)
                        print(email_body)
                        print(name)
                        print(f"E-mails trouvés dans le fichier {temp_paths[i]} : {emails}")

                        sender_email = "recruitb18@gmail.com"
                        sender_password = "your key app"

                        try:
                            server = smtplib.SMTP('smtp.gmail.com', 587)
                            server.starttls()
                            server.login(sender_email, sender_password)

                            for email in emails:
                                msg = MIMEMultipart()
                                msg['From'] = sender_email
                                msg['To'] = email
                                msg['Subject'] = email_subject
                                msg.attach(MIMEText(email_body, 'plain'))

                                server.sendmail(sender_email, email, msg.as_string())
                                print(f"E-mail envoyé à {email} depuis le fichier {temp_paths[i]}")
                                emails_sent += 1
                                sent_emails_list.append(email)  # Ajouter l'e-mail à la liste

                            server.quit()
                        except Exception as e:
                            print(f"Erreur lors de l'envoi des e-mails pour le fichier {temp_paths[i]} : {e}")

                # Supprimer les fichiers temporaires
                for temp_path in temp_paths:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                # Générer une réponse avec Mistral
                try:
                    if sent_emails_list:
                        prompt = (
                            "Vous êtes un assistant spécialisé dans le recrutement. "
                            f"{emails_sent} e-mails ont été envoyés avec succès aux candidats suivants : {', '.join(sent_emails_list)}. "
                            "Générez une phrase professionnelle et contextuelle pour informer l'utilisateur de cette action."
                        )
                    else:
                        prompt = (
                            "Vous êtes un assistant spécialisé dans le recrutement. "
                            "Aucun e-mail n'a été envoyé car aucun candidat ne correspond à la catégorie spécifiée ou aucun e-mail n'a été trouvé dans les CV. "
                            "Générez une phrase professionnelle pour informer l'utilisateur de cette situation."
                        )

                    response = llm.invoke(prompt)
                    response = response.split("\n\n")[1] if "\n\n" in response else response
                except Exception as e:
                    logger.error(f"Erreur lors de la génération de la réponse : {str(e)}")
                    response = f"{emails_sent} e-mails ont été envoyés avec succès."

                # Enregistrer la conversation dans la base de données
                Chat.objects.create(
                    user=user,
                    message=f"{message} - Sujet: {email_subject}, Corps: {email_body}",
                    response=response
                )

                return Response({
                    "status": "success",
                    "response": response,  # Utiliser Mistral pour générer le message
                    "emails": sent_emails_list  # Retourner la liste des e-mails envoyés
                }, status=status.HTTP_200_OK)
            elif intention == "ask_question":
                try:
                    if not cv_files:
                        # Si aucun fichier CV n'est fourni, traiter uniquement la question
                        prompt = (
                            "Vous êtes un assistant spécialisé dans le recrutement et l'analyse de CVs. "
                            f"Question : {message}\n\n"
                            "Répondez en vous basant uniquement sur le contexte du recrutement."
                        )
                        full_response = llm.invoke(prompt)
                        print(f"Réponse pour la question sans CV : {full_response}")

                        # Extraire la partie après le deuxième \n\n
                        parts = full_response.split("\n\n")
                        if len(parts) > 2:  # Vérifier qu'il y a au moins deux \n\n
                            response = "\n\n".join(parts[2:])  # Récupérer tout après le deuxième \n\n
                        else:
                            response = full_response  # Si moins de deux \n\n, retourner la réponse complète

                        # Enregistrer la réponse dans la base de données
                        Chat.objects.create(
                            user=user,
                            message=message,
                            response=response
                        )

                        return Response({
                            "status": "success",
                            'response': response
                        }, status=status.HTTP_200_OK)

                    else:
                        # Si des fichiers CV sont fournis, les traiter
                        temp_paths = []
                        for cv_file in cv_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                                for chunk in cv_file.chunks():
                                    temp_file.write(chunk)
                                temp_paths.append(temp_file.name)

                        resume_texts = extract_text_from_pdfs(temp_paths)

                        try:
                            prompt = (
                                "Vous êtes un assistant spécialisé dans le recrutement et l'analyse de CVs. "
                                "Voici le texte d'un CV :\n"
                                f"{resume_texts}\n"
                                f"Question : {message}\n"
                                "Répondez en vous basant uniquement sur les informations du CV et dans le contexte du recrutement."
                            )
                            full_response = llm.invoke(prompt)
                            print(f"Réponse pour le fichier CV : {full_response}")

                            response = full_response.split("\n\n")[1:]

                            # Supprimer les fichiers temporaires
                            for temp_path in temp_paths:
                                os.remove(temp_path)

                            Chat.objects.create(
                                user=user,
                                message=message,
                                response=response
                            )

                            return Response({
                                "status": "success",
                                'response': response
                            }, status=status.HTTP_200_OK)

                        except Exception as e:
                            for temp_path in temp_paths:
                                os.remove(temp_path)

                            return Response({
                                "status": "error",
                                "message": f"Erreur lors du traitement de la question : {str(e)}"
                            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                except Exception as e:
                    return Response({
                        "status": "error",
                        "message": f"Erreur : {str(e)}"
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            else:
                return Response({
                    "status": "error",
                    "message": "Intention non reconnue. Veuillez reformuler votre demande."
                }, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({
                "status": "error",
                "message": f"Erreur lors de la détection de l'intention : {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Vue pour récupérer les messages récents
@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_recent_chats_api(request):
    user = request.user
    one_hour_ago = timezone.now() - timedelta(minutes=60)
    recent_chats = Chat.objects.filter(user=user, created_at__gte=one_hour_ago)

    chats_data = [
        {
            'id': chat.id,
            'user': chat.user.username,
            'message': chat.message,
            'response': chat.response,
            'created_at': chat.created_at
        }
        for chat in recent_chats
    ]

    return Response({
        'status': 'success',
        'chats': chats_data
    }, status=200)
from rest_framework.views import APIView
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .models import CV 
# Vue basée sur une classe pour supprimer un CV

class CVDeleteAPIView(APIView):
    authentication_classes = [TokenAuthentication]  # Ajoutez cette ligne
    permission_classes = [IsAuthenticated]  # Gardez cette ligne

    def delete(self, request, cv_id):
        try:
            # Récupérer le CV par son ID
            cv = CV.objects.get(id=cv_id)
            # Supprimer le fichier associé du dossier `media`
            file_path = os.path.join(settings.MEDIA_ROOT, cv.path)
            if os.path.exists(file_path):
                os.remove(file_path)
            # Supprimer le CV
            cv.delete()
            # Retourner une réponse de succès
            return Response({"status": "success", "message": "CV supprimé avec succès"}, status=status.HTTP_204_NO_CONTENT)
        except CV.DoesNotExist:
            # Retourner une erreur si le CV n'existe pas
            return Response({"status": "error", "message": "CV non trouvé"}, status=status.HTTP_404_NOT_FOUND)
# Vue basée sur une fonction pour lister tous les CV
@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def CVListAPI(request):
    # Récupérer tous les CV
    cvs = CV.objects.all()
    
    # Sérialiser les données
    serializer = CVSerializer(cvs, many=True)
    
    # Retourner la réponse
    return Response(serializer.data, status=status.HTTP_200_OK)
from rest_framework import serializers
from .models import CV
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
class CVSerializer(serializers.ModelSerializer):
    class Meta:
        model = CV
        fields = ['id', 'user', 'profile', 'path', 'upload_date']