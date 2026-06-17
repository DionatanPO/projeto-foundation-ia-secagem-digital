import os
import re
import gc
import json
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
        return cls._instance

    @property
    def rag_service(self):
        if not hasattr(self, '_rag_service') or self._rag_service is None:
            self._rag_service = RagService()
        return self._rag_service

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

            self._patch_chat_template()

            self._current_model_path = model_path
            device = "GPU" if n_gpu_layers != 0 else "CPU"
            logger.info("Modelo carregado na %s: %s", device, os.path.basename(model_path))
        except Exception as e:
            logger.error("Erro ao carregar o modelo: %s", e)
            import traceback
            traceback.print_exc()
            self._model = None

    def _patch_chat_template(self):
        if self._model is None:
            return
        template = self._model.metadata.get('tokenizer.chat_template')
        if not template or 'enable_thinking' not in template:
            return
        modified = template.replace(
            "{%- if enable_thinking is defined and enable_thinking is false %}",
            "{%- if True %}"
        ).replace(
            "{{- '<think>\\n' }}",
            "{{- '' }}"
        )
        eos_token_id = self._model.token_eos()
        bos_token_id = self._model.token_bos()
        eos_token = self._model._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""
        bos_token = self._model._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""

        from llama_cpp.llama_chat_format import Jinja2ChatFormatter, chat_formatter_to_chat_completion_handler
        formatter = Jinja2ChatFormatter(
            template=modified,
            eos_token=eos_token,
            bos_token=bos_token,
            stop_token_ids=[eos_token_id],
        )
        self._model._chat_handlers['chat_template.default'] = formatter.to_chat_handler()

    DEFAULT_SYSTEM_PROMPT = (
        "Você é um assistente técnico especializado no sistema de gestão de secagem 'Secagem Digital'. "
        "Sua função é analisar estritamente os dados JSON fornecidos pelo usuário na mensagem de prompt e "
        "responder de forma técnica, objetiva e estruturada conforme as instruções. "
        "Não invente dados. "
        "Não utilize emojis em hipótese alguma. "
        "Se a informação não estiver presente nos dados fornecidos, responda que o dado está indisponível."
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

    @staticmethod
    def _strip_think(text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def generate_stream(self, prompt, temperature=0.1, image_base64=None, system_prompt=None, history=None, use_rag=False):
        """
        Streaming padronizado (NDJSON):
        {"event": "thought", "data": "..."}
        {"event": "message", "data": "..."}
        {"event": "metrics", "data": {...}}
        {"event": "done", "data": null}
        """
        if self._model is None:
            yield json.dumps({"event": "error", "data": "Modelo não carregado."}) + "\n"
            return

        import time
        start_time = time.time()
        first_token_time = None
        token_count = 0

        try:
            rag_context = self.rag_service.retrieve_context(prompt) if use_rag else ""
            text_content = f"CONTEXTO:\n{rag_context}\n\nPERGUNTA: {prompt}" if rag_context else prompt

            user_content = text_content
            if image_base64:
                user_content = [
                    {"type": "text", "text": text_content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]

            messages = [{"role": "system", "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT}]
            if history: messages.extend(history)
            messages.append({"role": "user", "content": user_content})

            stream = self._model.create_chat_completion(
                messages=messages, max_tokens=None, temperature=temperature, stream=True
            )

            in_think = False
            for chunk in stream:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        if first_token_time is None: first_token_time = time.time()
                        token_count += 1
                        content = delta['content']
                        
                        if not in_think and "<think>" in content:
                            in_think = True
                            content = content.replace("<think>", "")
                        
                        if in_think:
                            if "</think>" in content:
                                in_think = False
                                content = content.replace("</think>", "")
                                yield json.dumps({"event": "thought", "data": content}) + "\n"
                            else:
                                yield json.dumps({"event": "thought", "data": content}) + "\n"
                        else:
                            yield json.dumps({"event": "message", "data": content}) + "\n"

            metrics = {
                "tps": token_count / (time.time() - first_token_time) if first_token_time else 0,
                "tokens": token_count,
                "duration": time.time() - start_time
            }
            yield json.dumps({"event": "metrics", "data": metrics}) + "\n"
            yield json.dumps({"event": "done", "data": None}) + "\n"

        except Exception as e:
            yield json.dumps({"event": "error", "data": str(e)}) + "\n"

    def generate_response(self, prompt, temperature=0.1, image_base64=None, system_prompt=None, history=None, use_rag=False):
        """
        Envia o prompt formatado para o modelo e retorna a resposta gerada.
        """
        if self._model is None:
            return "Erro: O modelo LMM não está carregado."

        try:
            rag_context = ""
            if use_rag:
                rag_context = self.rag_service.retrieve_context(prompt)

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

            response_text = self._strip_think(output['choices'][0]['message']['content'])

            if response_text.startswith("ASSISTANT:"):
                response_text = response_text.replace("ASSISTANT:", "", 1).strip()
            if "USER:" in response_text:
                response_text = response_text.split("USER:")[0].strip()

            return response_text
        except Exception as e:
            return f"Erro na inferência: {str(e)}"
