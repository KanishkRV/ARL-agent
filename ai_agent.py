from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import speech_recognition as sr
import pyttsx3
import os
import re

from langchain_core.messages import HumanMessage, AIMessage

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

parser = PydanticOutputParser(pydantic_object=Researchresponse)

tools = []

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful research assistant.
            Your task is to answer user queries by conducting research and using the available tools.
            When you provide your final answer, it MUST be a JSON object that strictly conforms to the Researchresponse schema.
            DO NOT include any internal thoughts, conversational text, or any other characters outside the JSON object in your final response.
            The 'summary' should be a concise and informative overview of the research. 
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

agent = create_tool_calling_agent(
    llm=llm1,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

chat_history = []

SpeakText("Hello! I'm your research assistant. How can I help you today?")
print("Starting conversation. Say 'exit' or 'quit' to end.")

while True:
    try:
        with sr.Microphone() as source2:
            SpeakText("I'm listening.")
            r.adjust_for_ambient_noise(source2, duration=0.5)
            print("Listening...")
            audio2 = r.listen(source2, timeout=5, phrase_time_limit=10)

            MyText = r.recognize_google(audio2)
            query = MyText.lower()
            print(f"You said: {query}")

            if query in ["exit", "quit", "goodbye", "stop"]:
                SpeakText("Exiting chat. Goodbye!")
                print("Exiting chat. Goodbye!")
                break

            raw_agent_response = agent_executor.invoke(
                {"query": query, "chat_history": chat_history}
            )

            agent_output_text = raw_agent_response.get('output', "")
            print(f"\n--- Raw Agent Output (before cleaning) ---\n{agent_output_text}\n--- End Raw Output ---\n")

            try:
                json_match = re.search(r'\{.*\}', agent_output_text, re.DOTALL)

                if json_match:
                    clean_json_string = json_match.group(0)
                    print(f"--- Clean JSON String for Parsing ---\n{clean_json_string}\n--- End Clean JSON ---\n")
                    structured_response = parser.parse(clean_json_string)

                    speech_summary = f"Okay, I've researched {structured_response.topic}. Here's a summary: {structured_response.summary}. "
                    if structured_response.sources:
                        speech_summary += f"I found this information from sources like: {', '.join(structured_response.sources[:2])}. "
                    if structured_response.tools_used:
                        speech_summary += f"I used the following tools: {', '.join(structured_response.tools_used)}. "
                    else:
                        speech_summary += "I didn't need to use any special tools for this."

                    SpeakText(speech_summary)
                    print(f"Spoken response: {speech_summary}")

                    chat_history.append(HumanMessage(content=query))
                    chat_history.append(AIMessage(content=speech_summary))

                else:
                    raise ValueError("Agent's output did not contain a valid JSON object.")

            except Exception as parse_error:
                print(f"Error parsing agent output into Pydantic model: {parse_error}")
                fallback_message = "I couldn't format the research perfectly, but here's what I found: " + agent_output_text
                SpeakText(fallback_message)
                print(f"Fallback spoken response: {fallback_message}")

                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=agent_output_text))


    except sr.UnknownValueError:
        SpeakText("Sorry, I didn't catch that. Could you please repeat your query?")
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        SpeakText("My apologies, I'm having trouble connecting to the speech recognition service.")
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        SpeakText("An unexpected error occurred. Please try again or restart the assistant.")
        print(f"An unexpected error occurred: {e}")