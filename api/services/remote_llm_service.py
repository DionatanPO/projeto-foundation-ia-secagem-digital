import json
import logging
import time

try:
    import requests
    REQUESTS_INSTALLED = True
except ImportError:
    REQUESTS_INSTALLED = False

logger = logging.getLogger(__name__)

OPCODE_DEFAULT_SYSTEM_PROMPT = (
    "Você é um assistente técnico especializado no sistema de gestão de secagem 'Secagem Digital'. "
    "Sua função é analisar estritamente os dados JSON fornecidos pelo usuário na mensagem de prompt e "
    "responder de forma técnica, objetiva e estruturada conforme as instruções. "
    "Não invente dados. "
    "Não utilize emojis em hipótese alguma. "
    "Se a informação não estiver presente nos dados fornecidos, responda que o dado está indisponível."
)


class RemoteLLMService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RemoteLLMService, cls).__new__(cls)
        return cls._instance

    def _get_db_config(self):
        from ..models import RemoteConfig
        obj = RemoteConfig.objects.first()
        if obj is None:
            obj = RemoteConfig.objects.create()
        return obj

    def get_config(self):
        obj = self._get_db_config()
        return {
            "enabled": obj.enabled,
            "api_url": obj.api_url,
            "model": obj.model,
        }

    def set_config(self, config: dict):
        obj = self._get_db_config()
        obj.enabled = config.get("enabled", False)
        obj.api_url = config.get("api_url", "")
        obj.model = config.get("model", "")
        obj.save()

    def _base(self):
        return self.get_config()["api_url"].rstrip("/")

    def is_enabled(self):
        if not REQUESTS_INSTALLED:
            return False
        cfg = self.get_config()
        return cfg.get("enabled", False) and bool(cfg.get("api_url"))

    def test_connection(self):
        if not REQUESTS_INSTALLED:
            return {"success": False, "error": "Biblioteca 'requests' não instalada. Execute: pip install requests"}
        cfg = self.get_config()
        if not cfg.get("api_url"):
            return {"success": False, "error": "URL do OpenCode não configurada"}
        try:
            resp = requests.get(self._base() + "/global/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return {"success": True, "version": data.get("version", "desconhecida")}
            return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Timeout na conexão"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "Erro de conexão — OpenCode está rodando?"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_stream(self, prompt, temperature=0.1, image_base64=None, system_prompt=None, history=None, use_rag=False):
        if not REQUESTS_INSTALLED:
            yield json.dumps({"event": "error", "data": "Biblioteca 'requests' não instalada."}) + "\n"
            return
        if not self.is_enabled():
            yield json.dumps({"event": "error", "data": "OpenCode não configurado."}) + "\n"
            return

        start_time = time.time()
        token_count = 0
        first_token_time = None

        try:
            from .rag_service import RagService
            rag = RagService()
            rag_context = rag.retrieve_context(prompt) if use_rag else ""
            text_content = f"CONTEXTO:\n{rag_context}\n\nPERGUNTA: {prompt}" if rag_context else prompt

            full_prompt = text_content
            sys_msg = system_prompt or OPCODE_DEFAULT_SYSTEM_PROMPT
            full_prompt = f"{sys_msg}\n\n{text_content}"
            if history:
                historico = "\n".join(
                    f"{m['role']}: {m['content']}" for m in history[-6:]
                )
                full_prompt = f"{historico}\n\n{full_prompt}"

            session_id = self._create_session()
            if not session_id:
                yield json.dumps({"event": "error", "data": "Falha ao criar sessão no OpenCode."}) + "\n"
                return

            parts = [{"type": "text", "text": full_prompt}]
            if image_base64:
                parts.append({
                    "type": "file",
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "mime": "image/jpeg",
                    "filename": "image.jpg"
                })
            body = {"parts": parts}
            model_id = self.get_config().get("model", "").strip()
            if model_id:
                body["model"] = model_id

            resp = requests.post(
                self._base() + f"/session/{session_id}/message",
                json=body,
                timeout=300,
            )

            if resp.status_code != 200:
                yield json.dumps({"event": "error", "data": f"OpenCode retornou HTTP {resp.status_code}"}) + "\n"
                return

            data = resp.json()
            parts = data.get("parts", [])

            for part in parts:
                if part.get("type") == "text":
                    chunk = part.get("text", "")
                    if chunk:
                        if first_token_time is None:
                            first_token_time = time.time()
                        token_count += 1
                        yield json.dumps({"event": "message", "data": chunk}) + "\n"

            elapsed = time.time() - start_time
            tps = token_count / (time.time() - first_token_time) if first_token_time else 0
            yield json.dumps({"event": "metrics", "data": {
                "tps": round(tps, 2),
                "tokens": token_count,
                "duration": round(elapsed, 2)
            }}) + "\n"
            yield json.dumps({"event": "done", "data": None}) + "\n"

        except requests.exceptions.Timeout:
            yield json.dumps({"event": "error", "data": "Timeout na requisição ao OpenCode."}) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "data": str(e)}) + "\n"

    def generate_response(self, prompt, temperature=0.1, image_base64=None, system_prompt=None, history=None, use_rag=False):
        collected = []
        for chunk in self.generate_stream(
            prompt=prompt,
            temperature=temperature,
            image_base64=image_base64,
            system_prompt=system_prompt,
            history=history,
            use_rag=use_rag
        ):
            try:
                packet = json.loads(chunk.strip())
                if packet.get("event") == "message":
                    collected.append(packet["data"])
                elif packet.get("event") == "error":
                    return f"Erro: {packet['data']}"
            except json.JSONDecodeError:
                pass
        return "".join(collected)

    def _create_session(self):
        try:
            resp = requests.post(self._base() + "/session", json={
                "title": "Secagem Digital AI"
            }, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("id")
            return None
        except Exception as e:
            logger.error("Erro ao criar sessão OpenCode: %s", e)
            return None
