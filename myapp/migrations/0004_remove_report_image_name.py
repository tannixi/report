# Generated by Django 4.2.11 on 2024-04-28 07:35

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("myapp", "0003_report_image_name_alter_report_id"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="report",
            name="image_name",
        ),
    ]
