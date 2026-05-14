# Secagem Digital AI 🌾🤖

Uma solução avançada de assistência técnica inteligente para o setor agrícola e industrial, focada no monitoramento de silos e otimização do processo de secagem de grãos. O projeto utiliza modelos de linguagem multimodal (LMM) executados localmente para garantir privacidade, baixa latência e alta disponibilidade.

## 🚀 Funcionalidades Principais

- **Inferência LMM Local**: Execução de modelos de ponta (como DeepSeek e Gemma) em formato GGUF via `llama-cpp-python`.
- **Chat Multimodal**: Suporte a visão computacional, permitindo que o usuário envie imagens de telemetria ou condições de grãos para análise da IA.
- **Raciocínio Profundo (Think Mode)**: Integração com modelos de reasoning (Chain of Thought) para resolução de problemas técnicos complexos.
- **Interface Premium**: Dashboard moderno com suporte a temas Claro e Escuro (conforme a preferência do sistema).
- **Gestão de Memória Singleton**: Carregamento inteligente de modelos para otimizar o uso de RAM/VRAM.
- **Streaming de Respostas**: Experiência de chat fluida com respostas geradas em tempo real (token por token).

## 🛠️ Tecnologias Utilizadas

- **Backend**: Python 3.14 + [Django Framework](https://www.djangoproject.com/)
- **Motor de IA**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (ES6+)
- **Formato de Modelos**: GGUF (Quantização para alta performance local)
- **Processamento de Dados**: PSUtil (Monitoramento de hardware)

## 📋 Pré-requisitos

Antes de começar, você precisará ter instalado:
- Python 3.10 ou superior
- Compilador C++ (para o llama-cpp-python)
- Modelos `.gguf` dentro da pasta `models/`

## 🔧 Instalação e Configuração

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/DionatanPO/projeto-foundation-ia-secagem-digital.git
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências da IA (Escolha o comando correto para o seu hardware):**

   O código da aplicação é universal e suporta qualquer placa de vídeo ou processador. A única diferença é o método de instalação da biblioteca `llama-cpp-python`:

   **Opção A: Aceleração Universal via Vulkan (Recomendado para AMD, NVIDIA e Intel Arc)**
   ```bash
   set CMAKE_ARGS="-DGGML_VULKAN=on"
   pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   pip install -r requirements.txt
   ```

   **Opção B: Máxima Performance NVIDIA (CUDA)**
   ```bash
   set CMAKE_ARGS="-DGGML_CUDA=on"
   pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
   pip install -r requirements.txt
   ```

   **Opção C: Nuvem ou CPU Pura (Máquinas sem placa de vídeo)**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente:**
   Crie um arquivo `.env` na raiz do projeto:
   ```env
   MODEL_PATH=./models/seu_modelo.gguf
   MMPROJ_PATH=./models/mmproj-f16.gguf  # Opcional (para visão)
   N_GPU_LAYERS=35  # Ajuste conforme sua GPU
   N_THREADS=4
   ```

5. **Inicie o servidor:**
   ```bash
   python manage.py runserver
   ```

## 🏗️ Estrutura do Projeto

- `/api`: Endpoints REST para inferência, streaming e gestão de modelos.
- `/api/services/lmm_service.py`: O "coração" do sistema, gerenciando o ciclo de vida do modelo local.
- `/model_ui`: Interface web responsiva e elegante.
- `/models`: Pasta destinada aos arquivos de pesos (.gguf) dos modelos de IA.

## 📄 Licença

Este projeto é desenvolvido para fins industriais e de monitoramento técnico.

---
Desenvolvido por [Dionatan Oliveira](https://github.com/DionatanPO)
