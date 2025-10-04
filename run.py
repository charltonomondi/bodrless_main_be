#!/usr/bin/env python
"""
Django server runner for Bodrless application
"""
import os
import sys
import django
from django.core.management import execute_from_command_line

def main():
    """Run Django development server"""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    django.setup()

    # Run migrations first
    print("Running database migrations...")
    execute_from_command_line(['manage.py', 'migrate', '--verbosity=1'])

    # Start development server
    print("Starting Django development server...")
    execute_from_command_line(['manage.py', 'runserver', '0.0.0.0:8000'])

if __name__ == '__main__':
    main()