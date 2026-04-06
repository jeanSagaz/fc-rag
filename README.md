## Tecnologias implementadas:

- python
- langchain
- postgreSQL + pgVector
- openAI
- rag
- docker

## Setup

Crie o arquivo '.env' conforme o arquivo '.env.example'.  

**1. Subir banco**  
```bash
# Exemplo de comando Bash para copiar
docker-compose up -d  
```

**2. Criando ambiente virtual**  
```bash
# Exemplo de comando Bash para copiar
python -m venv venv    
```

Acessando ambiente virtual no Windows:  
No Prompt de Comando (CMD):  
```bash
# Exemplo de comando Bash para copiar
venv\Scripts\activate  
```

No PowerShell:  
```bash
# Exemplo de comando Bash para copiar
.\venv\Scripts\Activate.ps1  
```

Acessando ambiente virtual no Windows:  
```bash
# Exemplo de comando Bash para copiar
source venv/bin/activate  
```

**3. Instalar dependências**  
```bash
# Exemplo de comando Bash para copiar
pip install -r requirements.txt  
```

**4. Configurar variáveis**  
Adicione sua chave do openAI no arquivo '.env':  
OPENAI_API_KEY=...  

**5. Ingestão do PDF**  
```bash
# Exemplo de comando Bash para copiar
python src\ingest.py  
```

**6. Rodar chat**  
```bash
# Exemplo de comando Bash para copiar
python src\chat.py  
```
