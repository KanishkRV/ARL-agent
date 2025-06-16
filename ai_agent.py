from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import speech_recognition as sr
import pyttsx3 

load_dotenv()

r = sr.Recognizer() 

def SpeakText(command):
    
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()

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
    
    SpeakText("lets chat")
        
    try:
        
        # use the microphone as source for input.
        with sr.Microphone() as source2:
             
            r.adjust_for_ambient_noise(source2, duration=0.2)
            
            audio2 = r.listen(source2)
            
            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            query = MyText
            raw_response = agent_executor.invoke({"query": query})
            
            if query.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break
            
    except sr.RequestError as e:
        print("Could not request results",{0}& format(e))
        
    except sr.UnknownValueError:
        print("unknown error occurred")
    

    try:
        structured_response = parser.parse(raw_response.get('output'))
        SpeakText(structured_response)
    except:
        print("error in parse")
    
    