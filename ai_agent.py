from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class Researchresponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm1 = ChatOllama(model="qwen3:0.6b", temperature=0)

parser = PydanticOutputParser(pydantic_object = Researchresponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools =[]

agent = create_tool_calling_agent(
    llm=llm1,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # For the verbose thing it is used for the thing process so if the team do not want set it to true

while True:
        
    query = input("let start chatting: ")
    raw_response = agent_executor.invoke({"query": query})
    
    if query.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break

    try:
        structured_response = parser.parse(raw_response.get('output'))
        print(structured_response)
    except:
        print("error in parse")