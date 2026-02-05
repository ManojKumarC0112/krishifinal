from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class CustomUserCreationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    last_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    email = forms.EmailField(max_length=254, required=True, help_text='Required for account recovery.')
    
    # ✅ NEW FIELDS for Personalization
    main_crops = forms.CharField(
        max_length=100, 
        required=False, 
        help_text='E.g., Rice, Wheat, Tomato (Separated by comma)',
        widget=forms.TextInput(attrs={'placeholder': 'Rice, Wheat, Tomato...'})
    )
    
    SOIL_CHOICES = [
        ('', 'Select Soil Type'),
        ('Alluvial', 'Alluvial Soil'),
        ('Black', 'Black Soil (Regur)'),
        ('Red', 'Red Soil'),
        ('Laterite', 'Laterite Soil'),
        ('Desert', 'Desert / Sandy Soil'),
        ('Mountain', 'Mountain / Forest Soil'),
        ('Saline', 'Saline / Alkaline Soil'),
        ('Peaty', 'Peaty / Marshy Soil'),
    ]
    soil_type = forms.ChoiceField(
        choices=SOIL_CHOICES, 
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta(UserCreationForm.Meta):
        model = User
        # Add the new fields to the list so they appear in form.cleaned_data
        fields = UserCreationForm.Meta.fields + ('first_name', 'last_name', 'email', 'main_crops', 'soil_type')