from django.urls import path
from . import views

app_name = 'auth'

urlpatterns = [
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.LogoutView.as_view(), name='logout'),
    path('me/', views.CurrentUserView.as_view(), name='current-user'),
    path('password/change/', views.ChangePasswordView.as_view(), name='password-change'),
    path('token/refresh/', views.TokenRefreshView.as_view(), name='token-refresh'),
    path('password/reset/', views.PasswordResetView.as_view(), name='password-reset'),
    path('password/reset/confirm/', views.PasswordResetConfirmView.as_view(), name='password-reset-confirm'),
    path('profile/', views.ProfileView.as_view(), name='profile'),
]
