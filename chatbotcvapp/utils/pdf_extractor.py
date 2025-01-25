import pdfplumber
import re
from typing import List, Dict, Any,Optional
from PyPDF2 import PdfReader
from datetime import datetime
class PDFCVExtractor:
    def __init__(self):
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean the extracted text by removing special characters and extra whitespace."""
        text = re.sub(r"\(cid:[^\)]+\)", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Optional[str]]:
        """Extract personal information from a LaTeX-generated PDF CV."""
        personal_info = {
            "name": None,
            "job_title": None,
            "phone": None,
            "email": None,
            "linkedin": None,
            "github": None,
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Get text from first page
                text = pdf.pages[0].extract_text()
                
                # Split text into lines and clean them
                lines = [self.clean_text(line) for line in text.split('\n') if line.strip()]
                
                # First non-empty line should be the name
                if len(lines) > 0:
                    personal_info["name"] = lines[0].strip()
                
                # Second non-empty line should be the job title
                if len(lines) > 1:
                    personal_info["job_title"] = lines[1].strip()
                
                # Join all lines for other extractions
                text = ' '.join(lines)
                
                # Email extraction
                email_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text)
                if email_match:
                    personal_info["email"] = email_match.group(1)
                
                # Phone extraction - Moroccan format
                phone_match = re.search(r"(?:\+212|0)\s*[67]\s*(?:\d{2}\s*){4}", text)
                if phone_match:
                    personal_info["phone"] = phone_match.group().strip()
                
                # LinkedIn extraction - nouveau pattern
                linkedin_match = re.search(r"linkedin\.com/in/([^/\s]+)", text, re.IGNORECASE)
                if linkedin_match:
                    handle = linkedin_match.group(1).strip()
                    personal_info["linkedin"] = f"https://www.linkedin.com/in/{handle}"
                else:
                    # Pattern alternatif si le premier ne fonctionne pas
                    linkedin_match = re.search(r"@([^\s|]+)\s*\|", text)
                    if linkedin_match:
                        handle = linkedin_match.group(1).strip()
                        personal_info["linkedin"] = f"https://www.linkedin.com/in/{handle}"
                
                # GitHub extraction
                github_match = re.search(r"@([^\s\|]+)\s*(?=Compétences|$)", text)
                if github_match:
                    handle = github_match.group(1).strip()
                    personal_info["github"] = f"https://github.com/{handle}"
                
        except Exception as e:
            print(f"Error extracting information from PDF: {str(e)}")
            
        return personal_info

    def extract_skills_from_pdf(self, pdf_path: str) -> Dict[str, list]:
        """Extrait les compétences uniquement après la section 'Compétences' ou 'Compétences techniques'."""
        skills = {}
        section_found = False  # Indicateur pour savoir si la section "Compétences" a été trouvée

        # Liste des mots-clés pour exclure des sections non pertinentes (ex : Langues, Loisirs)
        exclude_keywords = [
            'langues', 'loisirs', 'éducation', 'expérience', 'projets',
            'cid', 'contact', 'email', 'linkedin', 'github', 'français', 'anglais', 'arabe'
        ]

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                # Vérifier si la section "Compétences" existe dans le texte de la page
                if not section_found:
                    # Recherche des mots "Compétences" ou "Compétences techniques"
                    if re.search(r'.*(compétences).*', text, re.IGNORECASE):
                        section_found = True

                # Après avoir trouvé la section, extraire les catégories et compétences
                if section_found:
                    # Recherche des catégories de compétences sous la forme "Nom de la catégorie : compétences"
                    category_pattern = re.compile(r"([A-Za-zÀ-ÿ\s]+)\s*:\s*(.*)")  # Catégories suivies de ":"
                    matches = category_pattern.findall(text)

                    for category, skills_list in matches:
                        category = category.strip()

                        # Si la catégorie contient des mots-clés d'exclusion, ignorez-la
                        if any(keyword.lower() in category.lower() for keyword in exclude_keywords):
                            continue
                        
                        # Ajouter la catégorie et ses compétences dans le dictionnaire
                        skills[category] = [skill.strip() for skill in skills_list.split(',')]
        
        return skills
    
  
    def extract_experiences_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extrait les expériences professionnelles du CV."""
        experiences = []
        section_found = False
        current_experience = None

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                lines = [self.clean_text(line) for line in page.extract_text().split("\n") if line.strip()]
                
                for line in lines:
                    # Détecter la section "Expérience professionnelle"
                    if not section_found:
                        if re.search(r"expérience professionnelle", line, re.IGNORECASE):
                            section_found = True
                            continue

                    # Détecter la fin de la section (début d'une nouvelle)
                    if section_found and re.search(r"éducation|projets principaux|langues|loisirs", line, re.IGNORECASE):
                        section_found = False
                        if current_experience:
                            experiences.append(current_experience)
                        current_experience = None
                        break

                    # Si dans la section expérience
                    if section_found:
                        # Détecter les expériences (poste + dates)
                        match_experience = re.match(r"^(.+?)\s+\((.*?)\)\s+([A-Za-zéû\-]+\s\d{4})\s+-\s+([A-Za-zéû\-]+.*)$", line)
                        if match_experience:
                            if current_experience:
                                experiences.append(current_experience)
                            current_experience = {
                                "poste": match_experience.group(1).strip(),
                                "date_debut": match_experience.group(3).strip(),
                                 "date_fin": self.get_current_date() if re.search(r"présent|en cours", match_experience.group(4), re.IGNORECASE) else match_experience.group(4).strip(),
                              
                                "entreprise": "",
                                "responsabilités": []
                            }
                            continue

                        # Si entreprise (ligne suivante après le poste)
                        if current_experience and not current_experience["entreprise"]:
                            current_experience["entreprise"] = line.strip()
                            continue

                        # Ajouter les responsabilités (lignes commençant par des puces)
                        if current_experience and (line.startswith("•") or line.startswith("-")):
                            current_experience["responsabilités"].append(line.lstrip("•- ").strip())

        # Ajouter la dernière expérience si elle existe
        if current_experience:
            experiences.append(current_experience)

        return experiences
    
    @staticmethod
    def get_current_date() -> str:
        """Retourne la date actuelle au format 'mois année'."""
        return datetime.now().strftime("%B %Y")

    def extract_education_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extrait la section Éducation ou Académique d'un CV en PDF.
        Remplace "Présent" ou "en cours" par la date actuelle.

        :param pdf_path: Chemin vers le fichier PDF.
        :return: Liste de dictionnaires contenant les informations sur les diplômes.
        """
        educations = []
        section_found = False
        current_education = None

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extraire et nettoyer les lignes de texte
                lines = [self.clean_text(line) for line in page.extract_text().split("\n") if line.strip()]
                
                for line in lines:
                    # Détecter la section "Éducation" ou "Académique"
                    if not section_found:
                        if re.search(r"éducation|académique", line, re.IGNORECASE):
                            section_found = True
                            continue

                    # Détecter la fin de la section (début d'une nouvelle)
                    if section_found and re.search(r"expérience|projets principaux|langues|loisirs", line, re.IGNORECASE):
                        section_found = False
                        if current_education:
                            educations.append(current_education)
                        current_education = None
                        break

                    # Si dans la section éducation
                    if section_found:
                        # Détecter les diplômes avec les dates (Ex: "Master en IoT et Big Data janvier 2024 - Présent")
                        match_degree = re.match(r"^(.*?)\s+([A-Za-zéû\-]+\s\d{4})\s*-\s*([A-Za-zéû\-]+.*)$", line)
                        if match_degree:
                            if current_education:
                                educations.append(current_education)
                            current_education = {
                                "diplome": match_degree.group(1).strip(),
                                "date_debut": match_degree.group(2).strip(),
                                "date_fin": self.get_current_date() if re.search(r"présent|en cours", match_degree.group(3), re.IGNORECASE) else match_degree.group(3).strip(),
                                "universite": ""
                            }
                            continue

                        # Si université (ligne suivante après le diplôme)
                        if current_education and not current_education["universite"]:
                            current_education["universite"] = line.strip()

        # Ajouter la dernière entrée si elle existe
        if current_education:
            educations.append(current_education)

        return educations
    def extract_projects_from_pdf(self,pdf_path):
        """
        Extrait les projets et leurs années à partir d'un CV en PDF.

        Args:
            pdf_path (str): Le chemin du fichier PDF.

        Returns:
            list: Une liste de dictionnaires contenant les informations des projets.
                Exemple : [{'nom': 'Chatbot intelligent pour la gestion des CV', 'annee': 2025}, ...]
        """
        projects = []

        # Extraire le texte du PDF
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pdf_text = ''
                for page in pdf.pages:
                    pdf_text += page.extract_text()
        except Exception as e:
            print(f"Erreur lors de la lecture du PDF : {e}")
            return projects

        # Rechercher la section "Projets" dans le texte
        section_pattern = r"(Projets principaux.*?)(?:\n{2,}|$)"  # Trouver la section "Projets principaux"
        section_match = re.search(section_pattern, pdf_text, re.DOTALL | re.IGNORECASE)

        if not section_match:
            print("Section 'Projets principaux' non trouvée.")
            return projects  # Retourner une liste vide si la section n'est pas trouvée

        projects_section = section_match.group(1)  # Contenu de la section "Projets principaux"
        print("Section 'Projets principaux' trouvée :", projects_section)

        # Rechercher chaque projet et son année dans la section
        project_pattern = r"(.+?)\s+(\d{4})"
        project_matches = re.findall(project_pattern, projects_section)

        if not project_matches:
            print("Aucun projet trouvé avec le motif spécifié.")
            return projects

        for match in project_matches:
            project_name, year = match
            projects.append({
                'nom': project_name.strip(),
                'annee': int(year.strip())
            })
        
        return projects
            
    def extract_languages_from_pdf(self, pdf_path: str):
        """
        Extrait les langues et leurs niveaux à partir d'un PDF contenant un CV en LaTeX formaté.

        Args:
            pdf_path (str): Le chemin du fichier PDF.

        Returns:
            list: Une liste de dictionnaires contenant deux champs : 'langue' et 'niveau'.
        """
        languages_and_levels = []  # Initialiser la liste des langues et niveaux

        try:
            # Ouvrir le fichier PDF avec pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pdf_text = ""
                # Extraire tout le texte du PDF
                for page in pdf.pages:
                    pdf_text += page.extract_text()

                # Debug : Afficher le texte extrait
                print("Texte extrait du PDF :")
                print(pdf_text)

                # Identifier la section "Langues" dans le texte
                section_pattern = r"Langues(.*?)(?:\n\s*\w|\Z)"  # Rechercher la section "Langues"
                section_match = re.search(section_pattern, pdf_text, re.DOTALL | re.IGNORECASE)

                if not section_match:
                    print("Section 'Langues' non trouvée dans le texte extrait.")
                    return languages_and_levels  # Retourne une liste vide si la section n'est pas trouvée

                languages_section = section_match.group(1)  # Contenu de la section "Langues"
                print("Contenu de la section 'Langues' :")
                print(languages_section)

                # Motif pour extraire les langues et leurs niveaux
                language_pattern = r"•\s*(\w+)\s*:\s*([\w\s]+(?:\([^()]*\))?)"
                language_matches = re.findall(language_pattern, languages_section)

                # Debug : Afficher les correspondances trouvées
                print("Correspondances trouvées :", language_matches)

                # Traiter les langues et leurs niveaux extraits
                for match in language_matches:
                    language_name = match[0].strip()  # Exemple : "Français"
                    full_level = match[1].strip()  # Exemple : "Courant (C1)" ou "Langue maternelle"
                    
                    languages_and_levels.append({
                        'langue': language_name,
                        'niveau': full_level
                    })

        except Exception as e:
            print(f"Erreur lors de la lecture ou de l'analyse du PDF : {e}")

        return languages_and_levels
if __name__ == "__main__":
    pdf_path="C:\\Users\\adm19\\chatbotcv\\chatbotcvapp\\utils\\cv (6).pdf"
    extractor = PDFCVExtractor()
    cv_data = extractor.extract_from_pdf(pdf_path
         )
    print(cv_data)
    extractor = PDFCVExtractor()
    skills_data = extractor.extract_skills_from_pdf(
        pdf_path)
    print(skills_data)
    extractor = PDFCVExtractor()
    experiences_data = extractor.extract_experiences_from_pdf(pdf_path)
    print(experiences_data)

    extractor = PDFCVExtractor()
    education_data = extractor.extract_education_from_pdf(pdf_path)
    print(education_data)


    extractor = PDFCVExtractor()
    projets_data = extractor.extract_projects_from_pdf(pdf_path)
    print(projets_data)

    languages = extractor.extract_languages_from_pdf(pdf_path)
    print(languages)