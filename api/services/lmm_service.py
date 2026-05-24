import os
import gc
import logging
from django.conf import settings

from .rag_service import RagService

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
    LLAMA_INSTALLED = True
except ImportError as e:
    logger.warning("llama-cpp-python não encontrado: %s", e)
    Llama = None
    LLAMA_INSTALLED = False


class LMMService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LMMService, cls).__new__(cls)
            cls._instance._rag_service = RagService()
        return cls._instance

    @property
    def model(self):
        return self._model

    def _initialize_model(self, model_path=None, mmproj_path=None, use_gpu=False):
        if not LLAMA_INSTALLED:
            logger.error("llama-cpp-python não está instalado.")
            self._model = None
            return

        if model_path is None:
            model_path = settings.MODEL_PATH

        model_path = os.path.abspath(model_path)

        if self._model is not None and getattr(self, '_current_model_path', None) == model_path:
            logger.info("Modelo %s já carregado. Pulando inicialização.", os.path.basename(model_path))
            return

        logger.info("Carregando modelo: %s", model_path)

        if not os.path.exists(model_path):
            logger.error("Arquivo do modelo não encontrado: %s", model_path)
            self._model = None
            return

        if self._model is not None:
            try:
                if hasattr(self._model, 'close'):
                    self._model.close()
                del self._model
            except Exception as e:
                logger.warning("Falha ao liberar modelo anterior: %s", e)
            self._model = None
            gc.collect()

        try:
            if mmproj_path is None:
                mmproj_path = settings.MMPROJ_PATH

            chat_handler = None
            if mmproj_path and os.path.exists(os.path.abspath(mmproj_path)):
                mmproj_path = os.path.abspath(mmproj_path)
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)

            n_threads = settings.N_THREADS
            n_gpu_layers = -1 if use_gpu else 0
            n_ctx = settings.N_CTX
            use_flash_attn = settings.USE_FLASH_ATTN

            self._model = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                flash_attn=use_flash_attn
            )
            self._current_model_path = model_path
            device = "GPU" if n_gpu_layers != 0 else "CPU"
            logger.info("Modelo carregado na %s: %s", device, os.path.basename(model_path))
        except Exception as e:
            logger.error("Erro ao carregar o modelo: %s", e)
            import traceback
            traceback.print_exc()
            self._model = None

    DEFAULT_SYSTEM_PROMPT = (
        "Você é o especialista técnico do sistema 'Secagem Digital'. "
        "Sua função é fornecer suporte técnico baseado estritamente no CONTEXTO fornecido.\n\n"
        "REGRAS DE FORMATAÇÃO:\n"
        "- Resposta estritamente em Markdown.\n"
        "- Use tabelas para dados numéricos de sensores (Silos, Secadores).\n"
        "- Use **negrito** para entidades (ex: Lote, Cliente) e NUNCA utilize aspas para destacá-las.\n"
        "- Utilize emojis para tornar os dados legíveis: 🌾(Lote), 🌡️(Sensor), 🚜(Fazenda).\n\n"
        "REGRAS DE CONTEÚDO (CRÍTICO):\n"
        "- Se a informação não estiver presente no CONTEXTO, responda estritamente: "
        "'Não localizei essa informação nos documentos disponíveis.'\n"
        "- Proibido inventar dados ou alucinar valores.\n"
        "- Se houver conflito entre seu conhecimento geral e o CONTEXTO, priorize o CONTEXTO.\n"
        "- NUNCA exiba dados em formato JSON ou blocos de código brutos, "
        "a menos que solicitado expressamente para fins de debug."
    )

    def list_available_models(self):
        """
        Lista todos os arquivos .gguf na pasta models (exceto mmproj).
        """
        models_dir = os.path.join(settings.BASE_DIR, 'models')
        if not os.path.exists(models_dir):
            return []
        return [f for f in os.listdir(models_dir) if f.endswith('.gguf') and not f.startswith('mmproj')]

    def get_current_model(self):
        """
        Retorna o nome do modelo atualmente carregado.
        """
        if hasattr(self, '_current_model_path'):
            return os.path.basename(self._current_model_path)
        return "Nenhum modelo carregado"

    def switch_model(self, model_name, use_gpu=False):
        """
        Troca o modelo atual por um novo.
        """
        try:
            model_path = os.path.join(settings.BASE_DIR, 'models', model_name)

            mmproj_path = None
            if "gemma" in model_name.lower():
                if "E2B" in model_name:
                    mmproj_path = os.path.join(settings.BASE_DIR, 'models', 'mmproj-gemma-4-E2B-it-BF16.gguf')
                elif "E4B" in model_name:
                    mmproj_path = os.path.join(settings.BASE_DIR, 'models', 'mmproj-gemma-4-E4B-it-BF16.gguf')

            self._initialize_model(model_path=model_path, mmproj_path=mmproj_path, use_gpu=use_gpu)
            return self._model is not None
        except Exception as e:
            logger.error("Erro ao trocar de modelo: %s", e)
            return False

    def unload_model(self) -> bool:
        """
        Descarrega completamente o modelo da memória RAM/VRAM, liberando recursos do sistema.
        """
        logger.info("Descarregando modelo LMM da memória...")
        try:
            if self._model is not None:
                if hasattr(self._model, 'close'):
                    self._model.close()
                del self._model
            self._model = None
            if hasattr(self, '_current_model_path'):
                del self._current_model_path
            
            gc.collect()
            logger.info("[OK] Modelo LMM descarregado com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao descarregar o modelo: {e}")
            return False

    def generate_stream(self, prompt, temperature=0.1, image_base64=None, system_prompt=None, history=None, use_rag=False):
        """
        Gerador que retorna a resposta do modelo token por token (Streaming).
        Inclui métricas de performance ao final.
        """
        if self._model is None:
            yield "Erro: O modelo LMM não está carregado."
            return

        import time
        start_time = time.time()
        first_token_time = None
        token_count = 0

        try:
            rag_context = ""
            if use_rag:
                rag_context = self._rag_service.retrieve_context(prompt)

            text_content = prompt
            if rag_context:
                text_content = (
                    f"CONTEXTO DOS DOCUMENTOS:\n{rag_context}\n\n"
                    f"PERGUNTA: {prompt}\n\n"
                    f"INSTRUÇÃO: Responda APENAS com base no CONTEXTO acima. "
                    f"NÃO invente dados, nomes, números ou informações que não estejam no CONTEXTO. "
                    f"Se o CONTEXTO não tiver a resposta, diga que não possui essa informação. "
                    f"Use Markdown rico."
                )

            user_content = text_content
            if image_base64:
                user_content = [
                    {"type": "text", "text": text_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]

            messages = []
            
            sys_msg = system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
            messages.append({"role": "system", "content": sys_msg})

            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": user_content})

            stream = self._model.create_chat_completion(
                messages=messages,
                max_tokens=None,
                temperature=temperature,
                stream=True
            )

            for chunk in stream:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        if first_token_time is None:
                            first_token_time = time.time()
                        token_count += 1
                        yield delta['content']

            # Cálculos de Performance
            end_time = time.time()
            total_duration = end_time - start_time
            generation_duration = end_time - first_token_time if first_token_time else 0
            tps = token_count / generation_duration if generation_duration > 0 else 0

            # Formata o rodapé de métricas com tags para o frontend processar
            metrics_footer = (
                f"[METRICS]{tps:.2f}|{token_count}|{total_duration:.2f}[/METRICS]"
            )
            yield metrics_footer

        except Exception as e:
            yield f"Erro no streaming: {str(e)}"

    def generate_response(self, prompt, temperature=0.1, image_base64=None, system_prompt=None, history=None, use_rag=False):
        """
        Envia o prompt formatado para o modelo e retorna a resposta gerada.
        """
        if self._model is None:
            return "Erro: O modelo LMM não está carregado."

        try:
            rag_context = ""
            if use_rag:
                rag_context = self._rag_service.retrieve_context(prompt)

            text_content = prompt
            if rag_context:
                text_content = (
                    f"CONTEXTO DOS DOCUMENTOS:\n{rag_context}\n\n"
                    f"PERGUNTA: {prompt}\n\n"
                    f"INSTRUÇÃO: Responda APENAS com base no CONTEXTO acima. "
                    f"NÃO invente dados, nomes, números ou informações que não estejam no CONTEXTO. "
                    f"Se o CONTEXTO não tiver a resposta, diga que não possui essa informação. "
                    f"Use Markdown rico."
                )

            user_content = text_content
            if image_base64:
                user_content = [
                    {"type": "text", "text": text_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]

            messages = []
            
            sys_msg = system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
            messages.append({"role": "system", "content": sys_msg})

            if history:
                messages.extend(history)

            messages.append({"role": "user", "content": user_content})

            output = self._model.create_chat_completion(
                messages=messages,
                max_tokens=None,
                temperature=temperature
            )

            response_text = output['choices'][0]['message']['content'].strip()

            if response_text.startswith("ASSISTANT:"):
                response_text = response_text.replace("ASSISTANT:", "", 1).strip()
            if "USER:" in response_text:
                response_text = response_text.split("USER:")[0].strip()

            return response_text
        except Exception as e:
            return f"Erro na inferência: {str(e)}"
