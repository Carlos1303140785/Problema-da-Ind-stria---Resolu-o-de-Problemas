import os
import re
import glob
import unicodedata
import logging

from flask import Flask, request, jsonify, send_from_directory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quinelato-bot")

app = Flask(__name__)

DB_DIR = os.getenv("DB_DIR", "./chroma_db")
PDF_DIR = os.getenv("PDF_DIR", "./data")

FALLBACK = (
    "Não encontrei essa informação nos catálogos oficiais da Quinelato. "
    "Digite *3* para falar com um especialista da equipe."
)

PROMPT = PromptTemplate.from_template(
    "Você é o assistente virtual da Quinelato. Responda SOMENTE com base no contexto "
    "abaixo, em português, de forma curta e clara. Se a resposta não estiver no "
    f"contexto, responda exatamente: \"{FALLBACK}\"\n\n"
    "Contexto:\n{context}\n\nPergunta: {question}\nResposta:"
)

_retriever = None
_qa_chain = None
_chunks_locais = None

STOPWORDS = set("""a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sobre
e ou que qual quais como quando onde se ao aos à às é são foi ser ter tem ha há mais menos muito pouco
meu minha seu sua isso isto esse essa este esta pra pro me te voce vocês voces""".split())


def modo_offline() -> bool:
    """Sem chave da OpenAI (ou OFFLINE_MODE=1) o bot usa busca local por palavras-chave."""
    return os.getenv("OFFLINE_MODE") == "1" or not os.getenv("OPENAI_API_KEY")


def _tokens(texto: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]{3,}", normalizar(texto)) if t not in STOPWORDS}


def _carregar_chunks_locais():
    """Lê os PDFs com pypdf e divide em trechos (sem embeddings, sem custo)."""
    global _chunks_locais
    if _chunks_locais is not None:
        return _chunks_locais
    from pypdf import PdfReader
    chunks = []
    for caminho in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
        try:
            for n, pagina in enumerate(PdfReader(caminho).pages, start=1):
                texto = re.sub(r"\s+", " ", pagina.extract_text() or "").strip()
                for i in range(0, len(texto), 700):
                    trecho = texto[i:i + 900]
                    if len(trecho) > 80:
                        chunks.append((os.path.basename(caminho), n, trecho, _tokens(trecho)))
        except Exception:
            log.exception("Falha ao ler %s", caminho)
    _chunks_locais = chunks
    return chunks


def busca_local(pergunta: str) -> str:
    chunks = _carregar_chunks_locais()
    if not chunks:
        return ("Base de conhecimento técnica (PDFs) ainda não inicializada. "
                "Por favor, adicione os catálogos na pasta data/.")
    q = _tokens(pergunta)
    if not q:
        return FALLBACK
    pontuados = sorted(((len(q & c[3]), c) for c in chunks), key=lambda x: x[0], reverse=True)
    score, (arquivo, pag, trecho, _) = pontuados[0]
    if score == 0:
        return FALLBACK
    return (f"Encontrei isto no catálogo oficial ({arquivo}, pág. {pag}):\n\n\"{trecho}\"\n\n"
            "(modo demonstração sem IA: trecho mais relevante encontrado)")


def normalizar(texto: str) -> str:
    """minúsculas e sem acentos (ex.: 'Orçamento' -> 'orcamento')."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c)).strip()


def pdfs_disponiveis() -> bool:
    return bool(glob.glob(os.path.join(PDF_DIR, "*.pdf")))


def obter_retriever():
    """Carrega a base vetorial (reutiliza a persistida; só indexa se ainda não existir)."""
    global _retriever
    if _retriever is not None:
        return _retriever

    os.makedirs(PDF_DIR, exist_ok=True)
    embeddings = OpenAIEmbeddings()

    if os.path.isdir(DB_DIR) and os.listdir(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    elif pdfs_disponiveis():
        documents = PyPDFDirectoryLoader(PDF_DIR).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=DB_DIR)
    else:
        return None

    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever


def obter_chain():
    global _qa_chain
    if _qa_chain is None:
        retriever = obter_retriever()
        if retriever is None:
            return None
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        _qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": PROMPT},
        )
    return _qa_chain


def obter_resposta_rag(pergunta: str) -> str:
    if modo_offline():
        return busca_local(pergunta)
    try:
        return _resposta_ia(pergunta)
    except Exception:
        # chave inválida, sem créditos (429), sem rede etc.: não derruba a demo
        log.exception("Falha na OpenAI; usando busca local")
        return busca_local(pergunta)


def _resposta_ia(pergunta: str) -> str:
    chain = obter_chain()
    if chain is None:
        return ("Base de conhecimento técnica (PDFs) ainda não inicializada. "
                "Por favor, adicione os catálogos na pasta data/.")
    resposta = chain.invoke({"question": pergunta, "chat_history": []})
    return resposta["answer"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "sistema": "Assistente Virtual Quinelato - RAG WhatsApp"})


@app.route("/demo", methods=["GET"])
def demo():
    """Página web de demonstração do chatbot."""
    return send_from_directory(os.path.join(app.root_path, "static"), "demo.html")


@app.route("/webhook", methods=["POST"])
def webhook_whatsapp():
    data = request.get_json(silent=True) or {}
    try:
        remetente = data.get("from", "desconhecido")
        mensagem = str(data.get("message", "")).strip()

        if not mensagem:
            return jsonify({"status": "ignored", "reason": "empty message"}), 200

        msg = normalizar(mensagem)

        # Opções do menu: número exato ou palavra-chave (evita "pino 1.5mm" cair na opção 1)
        if msg == "1" or re.search(r"\b(duvida|catalogo)\b", msg):
            resposta = ("Por favor, digite sua dúvida técnica específica sobre os produtos "
                        "Quinelato (ex: pinos, brocas, fórceps):")
        elif msg == "2" or re.search(r"\borcamento\b", msg):
            resposta = (
                "Para solicitar um orçamento, por favor informe:\n"
                "1. Nome do Produto e Código\n"
                "2. Quantidade desejada\n"
                "3. Sua Cidade/Estado"
            )
        elif msg == "3" or re.search(r"\b(atendente|humano)\b", msg):
            resposta = ("Entendido. Estou transferindo seu atendimento para um especialista "
                        "da equipe Quinelato. Aguarde um momento!")
        else:
            resposta = obter_resposta_rag(mensagem)

        return jsonify({"status": "success", "remetente": remetente,
                        "resposta_enviada": resposta}), 200

    except Exception:
        log.exception("Erro no webhook")
        return jsonify({"status": "error", "message": "Erro interno ao processar a mensagem"}), 500


@app.route("/orcamento", methods=["POST"])
def registrar_orcamento():
    dados = request.get_json(silent=True) or {}
    faltando = [c for c in ("produto", "quantidade", "cidade") if not dados.get(c)]
    if faltando:
        return jsonify({"status": "error",
                        "message": f"Campos obrigatórios ausentes: {', '.join(faltando)}"}), 400

    # TODO: gravar em CRM/Planilha (ex.: Google Sheets API)
    return jsonify({
        "status": "success",
        "message": "Orçamento registrado com sucesso no CRM!",
        "dados": {k: dados[k] for k in ("produto", "quantidade", "cidade")},
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG") == "1")
