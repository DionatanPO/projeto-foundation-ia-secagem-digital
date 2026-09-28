@echo off
chcp 65001 >nul
echo Configurando ambiente MSVC x64...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "VULKAN_SDK=C:\VulkanSDK\1.4.357.0"
set "PATH=%VULKAN_SDK%\Bin;%PATH%"
set "CMAKE_ARGS=-DGGML_VULKAN=on"
echo Iniciando compilacao do llama-cpp-python com GGML_VULKAN=on...
"F:\Backup Dispositivos\Documentos\Projetos\Projeto-SECAGEMDIGITAL-AI\venv\Scripts\python.exe" -m pip install --force-reinstall --no-cache-dir llama-cpp-python --verbose
