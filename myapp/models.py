from django.db import models

class Report(models.Model):  # Class names should be capitalized
    id = models.BigAutoField(primary_key=True)  # Explicitly defining the 'id' field
    image_id = models.CharField(max_length=20)
    report = models.TextField()
# Assuming you want to keep this field

    def __str__(self):
        return self.image_name