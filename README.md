# Documentação da Solução - Assistente Virtual Quinelato (WhatsApp + RAG)

## 1. Visão Geral
Este repositório contém a documentação completa e o código-fonte da solução desenvolvida para o **Desafio Quinelato 2026**. 
O objetivo é estabelecer um canal digital eficiente para o consumidor final (com foco em jovens dentistas), superando a restrição de ausência de API no site e de equipe de TI dedicada, através da integração via **WhatsApp Business** com um assistente virtual baseado em **RAG (Retrieval-Augmented Generation)** utilizando os catálogos e FAQs oficiais em PDF da Quinelato.

---

## 2. Arquitetura da Solução

* **Canal de Entrada:** WhatsApp Business (integrado via Webhook com Provedor Oficial / Evolution API / Twilio), acessado por QR Code ou links diretos posicionados no Instagram, LinkedIn e site.
* **Motor de IA & RAG:** 
  * Processamento e divisão (chunking) dos catálogos técnicos em PDF.
  * Indexação vetorial com embeddings e recuperação semântica restrita estritamente aos documentos oficiais (evitando alucinações).
* **Gestão de Leads e Orçamentos:** Coleta estruturada de dados (Produto, Quantidade, Cidade) e registro automatizado.
* **Transbordo Humano:** Alerta em tempo real para a equipe de atendimento caso a dúvida não seja resolvida ou o cliente solicite intervenção humana.

---

## 3. Estrutura do Projeto
```
quinelato_app/
│
├── app.py                # Aplicação Flask (Webhook WhatsApp + RAG + Orçamento + Transbordo)
├── requirements.txt      # Dependências do projeto Python
├── Dockerfile            # Configuração de containerização
├── docker-compose.yml    # Orquestração de containers
├── static/demo.html      # Página web de demonstração do chatbot (http://localhost:5000/demo)
├── data/                 # Pasta para os catálogos e FAQs em PDF da Quinelato
│   └── README.md
└── README.md             # Esta documentação
```
