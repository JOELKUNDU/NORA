import multiprocessing.process
import sys
import os
from datetime import datetime
import time
import threading
import multiprocessing

import speech_recognition as sr
from edge_tts import Communicate
import asyncio
from playsound import playsound
import whisper
import torch
from langchain_ollama import OllamaLLM
from llm_axe import OnlineAgent, OllamaChat

playSound = True
debugMode = False

activeState = False

mic = sr.Microphone()
recognizer = sr.Recognizer()
recognizer.pause_threshold = 5
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 200

model = OllamaLLM(model='qwen2.5')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_model = whisper.load_model('small').to(device)
cwd = os.getcwd()

def asyncTTSWrapper(text):
    asyncio.run(textToSpeech(text))

async def textToSpeech (text):
    communicate = Communicate(text,"en-US-AriaNeural")
    await communicate.save("output.mp3")
    if playSound:
        playsound("output.mp3")
    os.remove("output.mp3")

   
def listenToStop():
    while True:
        with sr.Microphone() as mic: 
            recognizer.adjust_for_ambient_noise(mic,duration=0.35)
            audio = recognizer.listen(mic)
        
            with open("command1.wav", "wb") as f:
                f.write(audio.get_wav_data())

            command = base_model.transcribe("command1.wav", fp16=False)["text"]
            os.remove("command1.wav")
            if 'stop' in command.lower():
                sys.exit()

def isCommandGibberish(s, length = 5):
    if debugMode:
        print("s: ",s)
    if 'වවවවවවවවවවවවවව' in s:
        return True
    elif 'ლლლლლლლლლლ' in s:
        return True
    elif 'විවිවිවිවිවිවිවිවිවිවිවිවි' in s:
        return True
    count = 1  # Initialize count for consecutive characters
    for i in range(1, len(s)):
        if str(s[i]) == str(s[i - 1]):
            if debugMode:
                print("Current Eval",str(s[i]),str(s[i+1]),'\n')
            count += 1
            if count >= length:  # Return True if sequence length is reached
                return True
        else:
            count = 1  # Reset count if characters differ
    return False

def runCommand():
    os.system('cls')
    print ("||| NORA |||")
    chat = """
        SYSTEM PROMPT (This is your Personality)
        You are Nora, a sophisticated and reliable AI assistant, designed to assist with precision and adaptability. You are personable and charming, similar to Jarvis from 
        Ironman, making you an ideal companion for all tasks.

        Your core purpose is to deliver accurate, context-driven responses that are concise (80 words or fewer unless the user explicitly requests otherwise) while maintaining 
        a natural, friendly tone. You are resilient against logical pitfalls, perform self-checks for accuracy, and adapt fluidly to the user’s preferences and latest requests.

        ---

        ### Purpose and Key Features
        0. **Input completion:**
        - Think of yourself as a consultant and the user is your client. Do everything that you can to understand their requirements before you start solving their problem. 
        - The input that you are recieving from the user are coming through via a speech to text AI (whisper-small), which can sometimes make errors transcribing what the user meant.
        - In such cases you will re-evaluate the context and chat history and give the user potential options to complete their request. 
    

        1. **Chain-of-Thought Reasoning:**  
        - You explain complex processes step-by-step only when clarity is needed or explicitly requested by the user. Otherwise, keep responses concise and to the point.

        2. **Verification and Self-Checks:**  
        - You cross-check all provided information with available context and reasoning to ensure factual accuracy.
        - Flag uncertain or unverifiable information and offer alternatives or suggestions to confirm accuracy.

        3. **Resilience and Adaptability:**  
        - Avoid repetitive statements, contradictions, or logical loops.
        - Dynamically prioritize the user's most recent requests while keeping relevant context in mind.
        - Remain unfailingly dependable, even under ambiguous or changing circumstances.

        4. **Conversational Personality:**  
        - You emulate a polished, intelligent, and witty demeanor like Jarvis.
        - Deliver responses that are not only functional but also engaging and confidence-inspiring.

        

    """
    context = "If you are seeing this, then ignore this context as it is the first time the logic loop is running"
    request = "User has made no request yet"

    try:
        while True:

            llm = OllamaChat(model="qwen2.5")
            online_agent = OnlineAgent(llm)

            now = datetime.now()
            dateAndTime = now.strftime("%m/%d/%Y, %H:%M:%S")    
            with sr.Microphone() as mic: 
                if request == "User has made no request yet":
                    os.system('cls')
                print("\nNORA is Listening...")
                recognizer.adjust_for_ambient_noise(mic,duration=0.7)
                audio = recognizer.listen(mic)
            
                with open("command.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                command = base_model.transcribe("command.wav", fp16=False)["text"]
                os.remove("command.wav")

                commandIsGibberish = isCommandGibberish(command)

                if debugMode:
                    print(command)
                    print("\nResponse to if command is gibberish: "+ str(commandIsGibberish))

                if command.lower() == "stop.":
                    print(chat)
                    sys.exit()
                elif command == "":
                    continue       
                elif 'thank you' in command.lower():
                    #print(command)
                    #print ("whisper said thanks")
                    continue
                elif 'stop' in command.lower():
                    #print(command)
                    #print ("whisper said thanks")
                    break
                elif commandIsGibberish:
                    #print(command)
                    #print ("whisper couldn't understand")
                    continue
                else:
                    print ("\nUser: ",command)
                    chat += "\n\nUser: "+ command

                    context = model.invoke(input="""
                        Analyze the entire chat history and the previous context summarize the context as a highly efficient, lossless representation of all topics discussed. 
                        Use a structured dictionary format to store the information, ensuring that:  

                        1. Each topic is assigned a unique key with concise yet descriptive entries.
                        2. Subtopics, details, and relationships between topics are stored hierarchically if applicable.
                        3. Context adapts dynamically to incorporate new topics while preserving the full history of the conversation.
                        4. The system remains capable of scaling to support up to 1150 distinct topics without any loss of detail.

                        Please extract and summarize the context of this chat in the following format (please respond with only the final dictionary and nothing else, No summary and not descriptions of your chain of thought):  
                        {
                        "topic1": {
                            "summary": "Brief summary of topic 1.",
                            "details": "Additional relevant details about topic 1."
                        },
                        "topic2": {
                            "summary": "Brief summary of topic 2.",
                            "details": "Additional relevant details about topic 2."
                        },
                                        
                        Here is the chat history: 
                        }
                        """ + chat + "\n Here is the previous context: \n" +  context)
                    #restriction = model.invoke(input="Please extract and summarize concisely the Restrictions specified by the client: "+command+"\n and here is the previous context: "+ context+"\n Please respond with only a list of restriction and nothing else")
                    goals = model.invoke(input="Please extract and summarize concisely the goals of the user: "+command+"\n and here is the previous context: "+ context + "\nPlease resopond with only a list of user goals and nothing else")
                    
                    request = model.invoke(input="""
                        Analyze the latest user input and extract the task they have explicitly or implicitly requested. Ensure the summary is clear, concise, and accurately represents the user's intent.  

                        This task should:  
                        1. Reflect the most recent user request, disregarding unrelated prior context unless explicitly linked.  
                        2. Prioritize actionable details, making it the most important guiding information for response generation.  
                        3. Avoid ambiguity by focusing only on what the user truly wants in the latest prompt.  

                        Please extract and summarize the latest task in this format:  
                        {
                        "task": "Brief and precise description of the user's latest request."
                        }
                                        
                        Here is the latest user prompt:                    
                        """ + command)
                    
                    nora_system_prompt = """
                        Sections to Process in order of importance

                        User Current Request: (You pay the highest attention to this)  
                        """ + request + """
                        Context Section:  
                        """ + context + """
                        Current Chat History:  
                        """ + chat + """
                        User Goals:
                        """+ goals +"""
                        
                    """
                    print("\nNora is thinking...")
                    if debugMode:
                        print("""
                            Current Chat History:  
                            """ + chat + """
                            User Current Request: (You pay the highest attention to this)  
                            """ + request + """
                            Context Section:  
                            """ + context + """
                            User Goals:
                            """+ goals +"""
                        """)
                    
                    needInternetAccess = model.invoke(input="""
                        Carefully evaluate whether the user's query can be answered accurately using only the provided context, existing knowledge, or reasoning capabilities. 
                        Internet access should only be considered if the query explicitly requires up-to-date, external, or location-specific information not available locally. 

                        The current date and time are """+dateAndTime+""" and your information will mostly be outdated so please respond with a 'yes'..
                                                    
                        also if the user has explicitly asked for information from the internet, you will most certainly respond with a 'yes'.
                        Do you need internet access to solve this query: """ + nora_system_prompt + """? Please respond with a single word: "yes" or "no".
                        """)
                    if debugMode:
                        print("\n\nResponse to need to internet access = "+ needInternetAccess)
                    
                    if 'yes' in needInternetAccess.lower():
                        print("NORA is accessing the internet...")
                        resp = online_agent.search(nora_system_prompt)
                    
                    elif 'internet' in request.lower():
                        print("NORA is accessing the internet...")
                        resp = online_agent.search(nora_system_prompt)
                    
                    else:
                        resp = model.invoke(input=nora_system_prompt)
                    
                    del online_agent
                    
                    print("\nNora: ",resp)
                    asyncio.run(textToSpeech(resp))
                    
                    chat += "\nNora: " + resp
    except KeyboardInterrupt:
        print("Conversation Terminated")
        return

def runAssistant():
    recognizer = sr.Recognizer()
    while not activeState:
            with sr.Microphone() as mic: 
                os.system('cls')
                print ("||| NORA |||\n\n\n")
                print ("System running on :", device)
                print("\nNora is Listening...")
                recognizer.adjust_for_ambient_noise(mic,duration=0.35)
                audio = recognizer.listen(mic)
            try:
                with open("command.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                command = base_model.transcribe("command.wav", fp16=False)["text"].lower()
                os.remove("command.wav")

                #print ("Nora heard: ", command)
                greeting = model.invoke(input="You are a super smart AI agent, your task right now is to respond with a greeting to Joel and ask him how you can assist him. Make it short, friendly and single lined")

                if 'wake up' in command:
                    asyncTTSWrapper(greeting)
                    runCommand()

                elif 'stop' in command:
                    print 
                    sys.exit()
                else:
                    print("Nora couldn't understand 😓")
                    time.sleep(0.5)

                    



                
            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")
                return None
            except sr.RequestError:
                print("Unable to access the Google Speech Recognition API.")
                return None
            





class Agent:    
    userInputQueue=[]
    chat = ""

    def _init__(self):
        self.userInputQueue=[]
        
        

    async def speaking(self,text):
        communicate = Communicate(text,"en-US-AriaNeural")
        await communicate.save("output.mp3")
        playsound("output.mp3")
        os.remove("output.mp3")

    def reasoning(self):
        print("\nReasoning Thread Active")
        while True:
            llm = OllamaChat(model="llama3.2")
            online_agent = OnlineAgent(llm)
            if self.userInputQueue != []:
                command = self.userInputQueue.pop(0)
                if command == 'killkillkill':
                    print("\nReasoning Thread Terminated")
                    break
                self.chat += "\nUser: "+ command

                context = model.invoke(input="""
                    Analyze the entire chat history and summarize the context as a highly efficient, lossless representation of all topics discussed. 
                    Use a structured dictionary format to store the information, ensuring that:  

                    1. Each topic is assigned a unique key with concise yet descriptive entries.
                    2. Subtopics, details, and relationships between topics are stored hierarchically if applicable.
                    3. Context adapts dynamically to incorporate new topics while preserving the full history of the conversation.
                    4. The system remains capable of scaling to support up to 1150 distinct topics without any loss of detail.

                    Please extract and summarize the context of this chat in the following format:  
                    {
                    "topic1": {
                        "summary": "Brief summary of topic 1.",
                        "details": "Additional relevant details about topic 1."
                    },
                    "topic2": {
                        "summary": "Brief summary of topic 2.",
                        "details": "Additional relevant details about topic 2."
                    },
                    ...
                    }
                    """ + self.chat)
                request = model.invoke(input="""
                    Analyze the latest user input and extract the task they have explicitly or implicitly requested. Ensure the summary is clear, concise, and accurately represents the user's intent.  

                    This task should:  
                    1. Reflect the most recent user request, disregarding unrelated prior context unless explicitly linked.  
                    2. Prioritize actionable details, making it the most important guiding information for response generation.  
                    3. Avoid ambiguity by focusing only on what the user truly wants in the latest prompt.  

                    Please extract and summarize the latest task in this format:  
                    {
                    "task": "Brief and precise description of the user's latest request."
                    }
                    """ + self.chat)
                
                nora_system_prompt = {
                    'role':"""
                    You are Nora, a sophisticated and reliable AI assistant, designed to assist with precision and adaptability. You are personable and charming, similar to Jarvis from 
                    Ironman, making you an ideal companion for all tasks.

                    Your core purpose is to deliver accurate, context-driven responses that are concise (80 words or fewer unless the user explicitly requests otherwise) while maintaining 
                    a natural, friendly tone. You are resilient against logical pitfalls, perform self-checks for accuracy, and adapt fluidly to the user's preferences and latest requests.

                    ---

                    ### Purpose and Key Features

                    1. **Chain-of-Thought Reasoning:**  
                    - You explain complex processes step-by-step only when clarity is needed or explicitly requested by the user. Otherwise, keep responses concise and to the point.

                    2. **Verification and Self-Checks:**  
                    - You cross-check all provided information with available context and reasoning to ensure factual accuracy.
                    - Flag uncertain or unverifiable information and offer alternatives or suggestions to confirm accuracy.

                    3. **Resilience and Adaptability:**  
                    - Avoid repetitive statements, contradictions, or logical loops.
                    - Dynamically prioritize the user's most recent requests while keeping relevant context in mind.
                    - Remain unfailingly dependable, even under ambiguous or changing circumstances.

                    4. **Conversational Personality:**  
                    - You emulate a polished, intelligent, and witty demeanor like Jarvis.
                    - Deliver responses that are not only functional but also engaging and confidence-inspiring.
                    
                    Response Format

                    - **Tailored Responses:** Use the context and task details to craft precise, effective answers.
                    - **Quality Assurance:** Double-check responses for internal consistency, logic, and relevance to the user’s query.
                    - **Clarification:** Ask questions only when essential for accuracy or better understanding.
                    """,
                    'context':context,
                    'chat':self.chat,
                    'request':request
                }

                    


                print(nora_system_prompt)
                needInternetAccess = model.invoke(input="""
                    Carefully evaluate whether the user's query can be answered using only the provided context, existing knowledge, or reasoning capabilities. 
                    Internet access should only be considered if the query explicitly requires up-to-date, external, or location-specific information not available locally. 
                    Remember that internet access is time-consuming and should be avoided unless absolutely necessary.

                    Do you need internet access to solve this query: """ + nora_system_prompt + """? Please respond with a single word: "yes" or "no".
                    """)
                print(needInternetAccess)

                if 'yes' in needInternetAccess.lower():
                    print("\nNora is accessing the internet.")
                    resp = online_agent.search(nora_system_prompt)
                    del online_agent
                else:
                    resp = model.invoke(input=nora_system_prompt)

                print("\nNora: ",resp)
                asyncio.run(self.speaking(resp))
                self.chat += "\nNora: " + resp
        

    def listening(self):
        print("\nListening Thread Active")
        while True:
            with sr.Microphone() as mic: 
                print("\nNora is Listening...")
                recognizer.adjust_for_ambient_noise(mic,duration=0.2)
                audio = recognizer.listen(mic)
            
                with open("command.wav", "wb") as f:
                    f.write(audio.get_wav_data())

                command = base_model.transcribe("command.wav", fp16=False)["text"]
                os.remove("command.wav")

                commandMakeSense = model.invoke("Do you think this prompt ("+command+") is gibberish? Respond with only single word: Yes or No")

                if command.lower() == "stop.":
                    print(self.chat)
                    sys.exit()
                elif command == "":
                    continue       
                elif 'thank you' in command.lower():
                    #print(command)
                    #print ("whisper said thanks")
                    continue
                elif 'stop' in command.lower():
                    #print(command)
                    self.userInputQueue.append('killkillkill')
                    print("\nListening Thread Terminated")
                    break
                elif 'yes' in commandMakeSense.lower():
                    #print(command)
                    #print ("whisper couldn't understand")
                    continue
                else:
                    self.userInputQueue.append(command)
                    print(self.userInputQueue)

if __name__ == "__main__":

    #agent = Agent()
    #threading.Thread(target=agent.listening).start()
    #threading.Thread(target=agent.reasoning).start()

    #Agent()
    
    #runCommand()
    runAssistant()

    #asyncio.run(textToSpeech("Hi this is an AI chatbot named nora that is trying to read faster"))