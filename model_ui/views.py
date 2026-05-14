from django.shortcuts import render

def chat_interface(request):
    """
    Renderiza a interface frontend para testes do LMM.
    """
    return render(request, 'model_ui/index.html')
