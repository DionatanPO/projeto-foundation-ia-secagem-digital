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

Análise Estrutural do Projeto - Secagem Digital AI 🌾🤖
Com base na análise completa do código-fonte, aqui está o relatório detalhado:

1. Visão Geral
Projeto de Django REST API com interface web para sistema de assistência técnica agrícola/industrial focado em monitoramento de silos e secagem de grãos.

2. Arquitetura Técnica
Backend (Django + Python)
Framework: Django 3.x (versão antiga)
Linguagem: Python 3.14
API Framework: REST Framework
Servidor WSGI: WSGI Application padrão
Frontend
HTML5/CSS3/Vanilla JS (ES6+)
Template Único: model_ui/index.html
Interface Responsiva com suporte a temas Claro/Escuro
3. Componentes Principais
📁 /api/ - Endpoints REST
Endpoint	Método	Descrição
health/	GET	Status da API Django
chat/	POST	Inferência LMM com prompt
chat-stream/	POST	Streaming de resposta em tempo real
status/	GET	Consumo de memória RAM do sistema
models/	GET	Lista modelos disponíveis (.gguf)
switch-model/	POST	Troca de modelo LMM
clear-rag/	POST	Limpeza e reconstrução RAG
unload-model/	POST	Descarrega modelo da memória
📁 /model_ui/ - Interface Web
Renderização de chat interface
Suporte a múltiplos modelos LMM
Histórico de conversas
Métricas de performance em tempo real
4. Serviços Chave
LMMService (api/services/lmm_service.py)
Python

Apply
✅ Singleton Pattern - Carregamento único do modelo
✅ GPU/CPU Detection - Detecta hardware baseado em n_gpu_layers
✅ Flash Attention - Suporte a otimizações de atenção
✅ Model Switching - Troca dinâmica entre modelos GGUF
✅ Streaming Response - Token por token com métricas
Características:

Carrega modelos .gguf via llama-cpp-python
Suporta modelos: DeepSeek, Gemma (E2B/E4B)
Otimização de memória com garbage collection
Detecção automática de GPU/CPU
RagService (api/services/rag_service.py)
Python

Apply
✅ Vector Store Index - Embeddings HuggingFace
✅ Multi-Query Decomposition - Processamento de queries complexas
✅ Context Retrieval - Busca contextualizada
✅ Document Cleaning - Remove ruído (números de página)
✅ Persistence - Salva índice em JSON persist_dir
Embedding Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

5. Configuração do Projeto
Settings (core/settings.py)
Database: SQLite (db.sqlite3)
CORS: Permitido para todos os origins
Temas: Suporte a claro/escuro via JS
Autenticação: Django Auth padrão
Variáveis de Ambiente
Env

Apply
MODEL_PATH=./models/seu_modelo.gguf
MMPROJ_PATH=./models/mmproj-f16.gguf  # Opcional (visão)
N_THREADS=4
N_CTX=16384
USE_FLASH_ATTN=True
DEBUG=True
SECRET_KEY=<gerado>
6. Estrutura de Arquivos

Apply
├── api/                          # Endpoints REST
│   ├── admin.py                 # Django Admin (vazio)
│   ├── apps.py                  # App configuration
│   ├── models.py                # Modelos Django (vazio)
│   ├── serializers.py           # Rest Framework serializers
│   ├── services/
│   │   ├── lmm_service.py       # Core LMM engine
│   │   └── rag_service.py       # RAG vector store
│   ├── urls.py                  # URL routing
│   └── views.py                 # View functions
├── core/                         # Django configuration
│   ├── settings.py              # Settings file
│   ├── urls.py                  # Main URLs
│   └── wsgi.py                  # WSGI app
├── model_ui/                     # Frontend web
│   ├── admin.py                 # Django Admin (vazio)
│   ├── apps.py                  # App configuration
│   ├── models.py                # Modelos Django (vazio)
│   ├── static/
│   │   ├── css/                 # Stylesheets
│   │   └── js/                  # JavaScript files
│   ├── templates/               # HTML templates
│   └── tests.py                 # Unit tests
├── documents/                   # Documentos para RAG
├── models/                      # Modelos GGUF (.gguf)
├── storage/                     # Persistencia do RAG (JSON)
├── requirements.txt             # Python dependencies
└── manage.py                    # Django management script
7. Tecnologias Utilizadas
Categoria	Ferramentas
Backend	Django 3.x, REST Framework, llama-cpp-python
Frontend	HTML5, CSS3, Vanilla JS (ES6+)
IA/ML	HuggingFace Transformers, Sentence-BERT
Hardware	GPU/CPU detection, Flash Attention
Storage	SQLite, JSON persistência
8. Pontos Fortes ✅
Performance: Inferência LMM local com GPU support
Privacidade: Modelos executados localmente (sem upload)
Streaming: Respostas em tempo real token por token
RAG Integration: Banco vetorial para contexto contextualizado
Otimização: Singleton pattern, garbage collection
Flexibilidade: Suporte a múltiplos modelos GGUF
9. Pontos de Atenção ⚠️
Django Version: Versão 3.x (antiga - suporte limitado)
Sem Autenticação Real: Apenas Django Auth básico
Frontend Não Testado: Interface não implementada no código-fonte
Modelos GGUF: Dependem de arquivos externos na pasta models/
10. Instalação e Uso
Bash
Run
# 1. Clone o projeto
git clone https://github.com/DionatanPO/projeto-foundation-ia-secagem-digital.git

# 2. Ativar ambiente virtual
python -m venv venv
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas configurações

# 5. Iniciar servidor
python manage.py runserver

# 6. Acessar interface
http://localhost:8000/api/health/
11. Integração RAG 🔄
O sistema combina:

Documentos: Pasta documents/ com PDFs/textos
Embeddings: HuggingFace Sentence-BERT
Contexto: Busca por query + retrieval de documentos
Streaming: Respostas enriquecidas com contexto relevante
12. Próximos Passos Recomendados 🚀
Implementar autenticação Django (Django REST Auth)
Criar templates HTML completos para frontend
Testar integração RAG com documentos reais
Adicionar suporte a múltiplos usuários
Implementar logging mais robusto
Documentação de modelos GGUF específicos

📋 Casos de Uso - Secagem Digital AI 🌾🤖
Visão Geral
Sistema de assistência técnica inteligente para monitoramento e otimização do processo de secagem de grãos em silos agrícolas.

1. Usuário Final (Agrícola/Industrial)
Caso de Uso 1: Solicitar Análise Multimodal
Descrição: O usuário envia uma imagem ou vídeo de telemetria para análise da IA
Ator: Operador de silo
Entradas: Imagem/Vídeo, Prompt de descrição
Saída: Resposta com diagnóstico e recomendações
Fluxo: Upload → Análise Multimodal → Geração de Relatório
Caso de Uso 2: Solicitar Análise com Contexto RAG
Descrição: O usuário envia uma query para buscar informações relevantes dos documentos históricos
Ator: Técnico de manutenção
Entradas: Query, Histórico de conversas anteriores
Saída: Resposta contextualizada com base em documentos
Fluxo: Query → Retrieval de Documentos → Contextualização → Resposta
Caso de Uso 3: Trocar Modelo LMM
Descrição: O usuário troca o modelo atual por um novo (DeepSeek, Gemma)
Ator: Administrador do sistema
Entradas: Nome do modelo, Preferência GPU/CPU
Saída: Sucesso na troca de modelo
Fluxo: Seleção → Carregamento → Atualização
Caso de Uso 4: Limpar Banco Vetorial RAG
Descrição: O usuário limpa o armazenamento do RAG para reconstrução completa
Ator: Administrador do sistema
Entradas: N/A (limpeza automática)
Saída: Sucesso na limpeza e reconstrução
Fluxo: Limpeza → Reconstrução → Validação
Caso de Uso 5: Descarregar Modelo da Memória
Descrição: O usuário descarrega o modelo atual para liberar recursos
Ator: Administrador do sistema
Entradas: N/A (limpeza automática)
Saída: Sucesso na descarga e liberação de memória
Fluxo: Descarga → Garbage Collection → Validação
2. Sistema (Backend/API)
Caso de Uso 6: Iniciar Servidor
Descrição: O servidor Django inicia e prepara o ambiente
Ator: Administrador do sistema
Entradas: N/A
Saída: Servidor rodando em port 8000
Fluxo: Configuração → Instalação → Inicialização
Caso de Uso 7: Carregar Modelo LMM
Descrição: O sistema carrega um modelo GGUF na memória
Ator: Sistema (Singleton)
Entradas: Nome do modelo, Preferência GPU/CPU
Saída: Modelo carregado com métricas de performance
Fluxo: Seleção → Carregamento → Detecção de Hardware
Caso de Uso 8: Gerar Resposta Multimodal
Descrição: O sistema gera uma resposta para o prompt do usuário
Ator: LMMService (Singleton)
Entradas: Prompt, Temperatura, Imagem Base64, Histórico, RAG Contexto
Saída: Resposta formatada com métricas de performance
Fluxo: Processamento → Geração Token por Token → Formatação
Caso de Uso 9: Gerar Streaming de Resposta
Descrição: O sistema envia a resposta em tempo real token por token
Ator: LMMService (Singleton)
Entradas: Prompt, Temperatura, Imagem Base64, Histórico, RAG Contexto
Saída: Fluxo contínuo de tokens com métricas
Fluxo: Streaming → Tokenização → Formatação
Caso de Uso 10: Listar Modelos Disponíveis
Descrição: O sistema lista todos os modelos GGUF disponíveis
Ator: LMMService (Singleton)
Entradas: N/A
Saída: Lista de modelos com status
Fluxo: Busca → Filtragem → Formatação
Caso de Uso 11: Trocar Modelo
Descrição: O sistema troca um modelo para outro
Ator: LMMService (Singleton)
Entradas: Nome do modelo, Preferência GPU/CPU
Saída: Sucesso ou erro na troca
Fluxo: Verificação → Carregamento → Atualização
Caso de Uso 12: Limpar RAG Storage
Descrição: O sistema limpa e reconstrói o banco vetorial
Ator: RagService (Singleton)
Entradas: N/A (limpeza automática)
Saída: Sucesso na limpeza e reconstrução
Fluxo: Limpeza → Reconstrução → Validação
Caso de Uso 13: Descarregar Modelo
Descrição: O sistema descarrega o modelo atual da memória
Ator: LMMService (Singleton)
Entradas: N/A (limpeza automática)
Saída: Sucesso na descarga e liberação de memória
Fluxo: Descarga → Garbage Collection → Validação
3. Interface Web (Frontend)
Caso de Uso 14: Acessar Dashboard
Descrição: O usuário acessa a interface principal do sistema
Ator: Usuário Final
Entradas: N/A
Saída: Tela inicial com opções de navegação
Fluxo: Login → Autenticação → Renderização
Caso de Uso 15: Iniciar Chat Multimodal
Descrição: O usuário inicia uma conversa multimodal com a IA
Ator: Usuário Final
Entradas: Prompt, Imagem Base64 (opcional)
Saída: Resposta formatada em tempo real
Fluxo: Upload → Análise → Geração de Texto
Caso de Uso 16: Iniciar Chat com Contexto RAG
Descrição: O usuário inicia uma conversa buscando informações do histórico
Ator: Usuário Final
Entradas: Query, Histórico de conversas anteriores
Saída: Resposta contextualizada
Fluxo: Query → Retrieval → Contextualização
Caso de Uso 17: Enviar Imagem para Análise
Descrição: O usuário envia uma imagem para análise multimodal
Ator: Usuário Final
Entradas: Imagem Base64
Saída: Resposta com diagnóstico visual
Fluxo: Upload → Análise Multimodal → Geração
Caso de Uso 18: Enviar Query para RAG
Descrição: O usuário envia uma query buscando informações do histórico
Ator: Usuário Final
Entradas: Query, Histórico de conversas anteriores
Saída: Resposta contextualizada
Fluxo: Query → Retrieval → Contextualização
Caso de Uso 19: Trocar Modelo LMM
Descrição: O usuário troca o modelo atual para outro
Ator: Usuário Final/Administrador
Entradas: Nome do modelo, Preferência GPU/CPU
Saída: Sucesso na troca
Fluxo: Seleção → Carregamento → Atualização
Caso de Uso 20: Limpar RAG Storage
Descrição: O usuário limpa o armazenamento do RAG
Ator: Usuário Final/Administrador
Entradas: N/A (limpeza automática)
Saída: Sucesso na limpeza e reconstrução
Fluxo: Limpeza → Reconstrução
Caso de Uso 21: Descarregar Modelo
Descrição: O usuário descarrega o modelo atual da memória
Ator: Usuário Final/Administrador
Entradas: N/A (limpeza automática)
Saída: Sucesso na descarga e liberação de memória
Fluxo: Descarga → Garbage Collection

graph TD
    A[Usuário Final] --> B{Tipo de Acesso}
    B -->|Chat Multimodal| C1[Upload Imagem]
    B -->|Chat RAG| C2[Query + Histórico]
    B -->|Trocar Modelo| C3[Seleção de Modelo]
    B -->|Limpar RAG| C4[Limpeza Automática]
    
    C1 --> D[LMMService: Carregar Modelo]
    C2 --> E[RagService: Buscar Contexto]
    C3 --> F[LMMService: Trocar Modelo]
    C4 --> G[LMMService: Descarregar Modelo]
    
    D --> H[Inferência LMM]
    E --> I[Retrieval de Documentos]
    F --> H
    G --> H
    
    H --> J[Formatação Resposta]
    I --> K[Contextualização]
    J --> L[Resposta Final]
    K --> L
    
    L --> M[Interface Web]

    Diagrama de Sequência
Caso: Iniciar Chat Multimodal
sequenceDiagram
    participant User
    participant API as Django REST API
    participant LMMService as LMMService
    participant Model as Modelo GGUF
    
    User->>API: POST /api/chat/ com prompt e imagem
    API->>LMMService: Gerar resposta
    LMMService->>Model: Carregar modelo
    Model-->>LMMService: Status OK
    LMMService->>User: Enviar tokens token por token