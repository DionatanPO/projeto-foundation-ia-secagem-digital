import os
import re
import logging
from django.conf import settings

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

logger = logging.getLogger(__name__)


class RagService:
    _instance = None
    _index = None

    def __new__(cls):
        """
        Padrão Singleton: garante que o índice RAG seja carregado apenas uma vez.
        """
        if cls._instance is None:
            cls._instance = super(RagService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.documents_dir = os.path.join(settings.BASE_DIR, 'documents')
            self.persist_dir = os.path.join(settings.BASE_DIR, 'storage')

            os.makedirs(self.documents_dir, exist_ok=True)
            os.makedirs(self.persist_dir, exist_ok=True)

            logger.info("Inicializando RagService...")

            try:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                Settings.node_parser = SentenceSplitter(chunk_size=1500, chunk_overlap=300)
                Settings.llm = None
            except Exception as e:
                logger.error(f"Erro ao configurar embeddings: {e}")

            self._load_or_build_index()

    # ------------------------------------------------------------------
    # Limpeza de ruído universal e segura para qualquer documento.
    # Remove apenas linhas vazias ou números de página isolados.
    # ------------------------------------------------------------------
    _NOISE_PATTERNS = [
        r"^\s*\d+\s*$",  # Apenas números de página isolados (ex: "   24   ")
    ]

    def _clean_text(self, raw_text: str) -> str:
        """
        Remove linhas identificadas como ruído e limpa espaços de forma segura.
        Mantém a integridade de todo o texto útil.
        """
        lines = raw_text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Ignora linhas totalmente vazias
            if not stripped:
                continue
            # Ignora linhas de ruído (ex: números de página isolados)
            if any(re.search(p, stripped) for p in self._NOISE_PATTERNS):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def _get_cleaned_documents(self) -> list:
        """
        Lê os documentos e aplica limpeza de ruído de PDF antes de indexar.
        """
        try:
            raw_docs = SimpleDirectoryReader(self.documents_dir).load_data()
            cleaned_docs = []
            for doc in raw_docs:
                cleaned_text = self._clean_text(doc.text)
                cleaned_docs.append(Document(text=cleaned_text, metadata=doc.metadata))
            logger.info(f"Pre-processamento concluido. {len(cleaned_docs)} documento(s) limpos.")
            return cleaned_docs
        except Exception as e:
            logger.error(f"Erro ao limpar documentos: {e}")
            return []

    def _load_or_build_index(self):
        """
        Carrega o banco vetorial do disco ou constrói um novo.
        """
        if not os.listdir(self.documents_dir):
            logger.warning(f"Nenhum documento em {self.documents_dir}. RAG sem contexto.")
            self._index = None
            return

        if os.path.exists(os.path.join(self.persist_dir, "docstore.json")):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                self._index = load_index_from_storage(storage_context)
                logger.info("[OK] Indice RAG carregado do disco.")
                return
            except Exception as e:
                logger.warning(f"Falha ao carregar indice salvo, recriando. Erro: {e}")

        logger.info("Construindo novo indice RAG...")
        try:
            documents = self._get_cleaned_documents()
            self._index = VectorStoreIndex.from_documents(documents)
            self._index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info("[OK] Indice RAG construido e salvo.")
        except Exception as e:
            logger.error(f"Erro ao construir indice RAG: {e}")
            self._index = None

    def reload_documents(self) -> bool:
        """
        Reconstrói o índice a partir dos documentos atuais (use após adicionar arquivos).
        """
        logger.info("Recarregando documentos do RAG...")
        try:
            if not os.listdir(self.documents_dir):
                logger.warning(f"Nenhum documento em {self.documents_dir}.")
                self._index = None
                return False

            documents = self._get_cleaned_documents()
            self._index = VectorStoreIndex.from_documents(documents)
            self._index.storage_context.persist(persist_dir=self.persist_dir)
            logger.info("[OK] Indice RAG recarregado.")
            return True
        except Exception as e:
            logger.error(f"Erro ao recarregar documentos: {e}")
            return False

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Busca os trechos mais relevantes usando Multi-Query Decomposition.
        Divide perguntas compostas em sub-queries e consolida os resultados
        sem duplicatas, usando node_id para deduplicação eficiente.
        """
        if not self._index:
            return ""

        try:
            retriever = self._index.as_retriever(similarity_top_k=top_k)

            # Divide a query composta em sub-queries focadas
            parts = re.split(r'[?\n]', query)
            sub_queries = [p.strip() for p in parts if len(p.strip()) > 8]
            if not sub_queries:
                sub_queries = [query]

            logger.info(f"Sub-queries geradas: {sub_queries}")

            # Usa node_id (UUID curto) para deduplicação eficiente de memória
            seen_ids: set = set()
            all_node_texts: list = []

            for sub_q in sub_queries:
                nodes = retriever.retrieve(sub_q)
                for node in nodes:
                    node_id = node.node.node_id
                    if node_id not in seen_ids:
                        seen_ids.add(node_id)
                        all_node_texts.append(node.node.text)

            if not all_node_texts:
                return ""

            return "\n---\n".join(all_node_texts)

        except Exception as e:
            logger.error(f"Erro ao recuperar contexto do RAG: {e}")
            return ""

    def clear_and_rebuild_storage(self) -> bool:
        """
        Apaga fisicamente os arquivos do índice RAG na pasta storage/ e
        reconstrói o índice na RAM e no disco a partir dos arquivos atuais.
        Usa estratégias resilientes para contornar travas de arquivo no Windows.
        """
        logger.info("Limpando e reconstruindo storage do RAG...")
        try:
            import gc
            # 1. Reseta o índice na RAM
            self._index = None
            
            # Força o garbage collection para liberar handles de arquivos abertos no Windows
            gc.collect()

            # 2. Apaga ou zera os arquivos em persist_dir
            if os.path.exists(self.persist_dir):
                for filename in os.listdir(self.persist_dir):
                    file_path = os.path.join(self.persist_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.unlink(file_path)
                            logger.info(f"Arquivo deletado com sucesso: {filename}")
                        except Exception as ex:
                            # Se estiver travado pelo Windows, abre em modo escrita para zerar o conteúdo
                            logger.warning(f"Falha ao deletar {filename} ({ex}). Tentando zerar o arquivo...")
                            try:
                                with open(file_path, 'w') as f:
                                    f.truncate(0)
                                logger.info(f"Arquivo zerado com sucesso: {filename}")
                            except Exception as ex2:
                                logger.error(f"Erro crítico: Não foi possível zerar o arquivo {filename}: {ex2}")
            
            # 3. Reconstrói o índice do zero com os novos documentos
            self._load_or_build_index()
            return True
        except Exception as e:
            logger.error(f"Erro ao limpar e reconstruir storage do RAG: {e}")
            return False
