# In main/admin.py

from django.contrib import admin
from .models import Prediction, UserProfile, VideoAnalysis # ✅ Add VideoAnalysis

# Register your models here.
admin.site.register(Prediction)
admin.site.register(UserProfile) # ✅ Register UserProfile
admin.site.register(VideoAnalysis) # ✅ Register the new VideoAnalysis model