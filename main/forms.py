# In main/forms.py (NEW FILE)

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class CustomUserCreationForm(UserCreationForm):
    # We add the fields we want here
    first_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    last_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    email = forms.EmailField(max_length=254, required=True, help_text='Required. Your email.')

    class Meta(UserCreationForm.Meta):
        # We start with the default model
        model = User
        # And add our new fields to the list
        fields = UserCreationForm.Meta.fields + ('first_name', 'last_name', 'email')