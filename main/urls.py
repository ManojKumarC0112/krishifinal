# main/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # --- Auth Views ---
    path("signup/", views.signup_view, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # --- Central Dashboard ---
    path("analyze-field/", views.analyze_field_dashboard_view, name="analyze_field_dashboard"),
    
    # --- Main App Views ---
    path("", views.index_view, name="index"),
    path("predict/", views.predict_view, name="predict"),
    path("live-vision/", views.live_vision_view, name="live_vision"),

    # --- VIDEO ANALYSIS PATHS ---
    # Standard upload view
    path("video-analysis/", views.video_analysis_upload_view, name="video_analysis_upload"),
    # ✅ MISSING LINK FIXED: The Instant Demo Route
    path("demo-analysis/", views.demo_analysis_view, name="demo_analysis"),
    
    path("history/", views.history_view, name="history"),
    path("result/<int:prediction_id>/", views.result_detail_view, name="result_detail"),
    path("video-result/<int:video_id>/", views.video_result_detail_view, name="video_result_detail"),
    
    # --- Utility Pages ---
    path("schemes/", views.schemes_view, name="schemes"),
    path("weather/", views.weather_view, name="weather"),
    path("chatbot/", views.chatbot_view, name="chatbot"),
    path("profile/", views.profile_view, name="profile"),
    path("personalize/", views.personalize_view, name="personalize"),
    path("market-prices/", views.crop_prices_view, name="crop_prices"),
    path("direct-sell/", views.direct_sell_view, name="direct_sell"),

    # --- API Endpoints ---
    path("api/chatbot/", views.chatbot_api, name="chatbot_api"),
    path("api/live-vision/", views.api_live_vision, name="api_live_vision"),
    path("api/generate-report/", views.api_generate_report, name="api_generate_report"),
    path("api/process-video/<int:video_id>/", views.api_process_video, name="api_process_video"),
]