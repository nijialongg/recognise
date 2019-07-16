from django.shortcuts import render


def home(request):
    return render(request, '界面.html')

