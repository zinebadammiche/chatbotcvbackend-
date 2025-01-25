from django.urls import path
from .views import (
    signup, user_login, user_logout, UploadCVAPIView,
    ChatbotAPIView, CVDeleteAPIView, CVListAPI, get_recent_chats_api
)

urlpatterns = [
    path('api/upload_cv/', UploadCVAPIView.as_view(), name='upload_cv'),
    path('api/signup/', signup, name='signup'),
    path('api/login/', user_login, name='login'),
    path('api/logout/', user_logout, name='logout'),
    path('api/chatbot/', ChatbotAPIView.as_view(), name='chatbot'),
    path('api/recent-chats/', get_recent_chats_api, name='recent_chats_api'),
    path('api/cvs/', CVListAPI, name='cv-list'),  # Utilisez la fonction CVListAPI
    path('api/cvs/delete/<int:cv_id>/', CVDeleteAPIView.as_view(), name='cv-delete'),  # Utilisez la classe CVDeleteAPIView
]