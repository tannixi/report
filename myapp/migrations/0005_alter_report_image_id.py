# Generated by Django 4.2.11 on 2024-04-28 08:06

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("myapp", "0004_remove_report_image_name"),
    ]

    operations = [
        migrations.AlterField(
            model_name="report",
            name="image_id",
            field=models.CharField(max_length=20),
        ),
    ]
