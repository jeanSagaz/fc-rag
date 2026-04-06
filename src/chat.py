import os
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from search import search_prompt

load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model='gemini-2.5-flash-lite',    
#     temperature=0
# )

llm = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0
)

# llm = init_chat_model(
#     model='gpt-5-nano',
#     model_provider='openai',
# )

template = '''
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
'''

prompt_template = PromptTemplate.from_template(
    template=template,
)

chat_prompt = ChatPromptTemplate([template])

def build_context(results):
    return '\n\n'.join([doc.page_content for doc, _ in results])    

def main():
    print('🤖 Chat iniciado (digite "sair" para encerrar)\n')

    while True:
        question = input('PERGUNTA: ')

        if question.lower() in ['sair', 'exit', 'quit']:
            break

        result = search_prompt(query=question, k=10)
        context = build_context(result)

        # prompt = template.format(
        #     contexto=context,
        #     pergunta=question
        # )

        # prompt = prompt_template.format(
        #     contexto=context,
        #     pergunta=question,
        # )

        prompt = chat_prompt.format_messages(
            contexto=context, 
            pergunta=question
        )

        response = llm.invoke(prompt)

        print('\nRESPOSTA:', response.content, '\n')

if __name__ == '__main__':
    main()
