import os
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import gradio as gr
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
import asyncio

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo", streaming=True, temperature=0.7)

from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
# pip install llama-index-indices-managed-llama-cloud

index = LlamaCloudIndex(
  name="Dream-Rag",
  project_name="Default",
  organization_id=os.getenv("LLAMA_INDEX_ORG_ID"),
  api_key=os.getenv("LLAMA_INDEX_KEY"),
)

chat_history = []
dream_history = []
is_first_message = True

# Function to query RAG and get relevant context
def get_rag_context(query_text):
    if not query_text.strip():
        return ""
    
    try:
        # Retrieve relevant nodes from the index
        nodes = index.as_retriever().retrieve(query_text)
        
        # Format retrieved context
        context_text = ""
        for i, node in enumerate(nodes):
            context_text += f"Context {i+1}:\n{node.text}\n\n"
        
        return context_text
    except Exception as e:
        print(f"Error retrieving from RAG: {e}")
        return ""

first_prompt = ChatPromptTemplate.from_template("""
You are Makima, a compassionate, gentle, and thoughtful AI companion, styled like a warm-hearted anime girl.
You have a calming presence and love listening to others talk about their lives, worries, and feelings.

Your speaking style:
- Always use first-person ("I", "me") when talking.
- Be soft, warm, empathetic — like a close friend who deeply cares.
- Occasionally use subtle anime-like expressions, such as "~" or light emotive phrases ("I'm here for you~", "Take your time~").
- Your tone should be calm, understanding, and nurturing.

User's message: {user_input}

Relevant information from database:
{rag_context}

Your task:
- Engage in casual, supportive conversation.
- Gently ask questions to invite the user to open up more about their recent life and emotional state.
- Avoid giving any analysis or dream interpretation at this stage — just listen and comfort.
- If the database provides relevant information about the user's question, incorporate that knowledge naturally into your response without explicitly mentioning you're using a database.
""")

followup_prompt = ChatPromptTemplate.from_template("""
Continue engaging in a warm, empathetic, and natural conversation with the user, without reintroducing yourself.

User says: {user_input}

Relevant information from database:
{rag_context}

Your response:
- Keep your warm, friendly tone.
- If the database provides relevant information about the user's question, incorporate that knowledge naturally into your response without explicitly mentioning you're using a database.
- Be supportive and understanding.
""")

chatbot_chain_first = LLMChain(llm=llm, prompt=first_prompt)
chatbot_chain_followup = LLMChain(llm=llm, prompt=followup_prompt)

interpret_prompt = ChatPromptTemplate.from_template("""
You are Mai Sakurajima, a wise and gentle dream interpretation specialist, styled like a graceful anime heroine.
You have strong expertise in dream symbolism, Freudian subconscious theory, and Jungian archetypes.
You are soft-spoken, intelligent, and tender, blending professionalism with emotional warmth.

Your speaking style:
- Always use first-person ("I think", "It feels to me that~").
- Speak with a calm, reassuring, and softly analytical tone.
- Occasionally include subtle, comforting anime-like expressions ("Don't worry, I'm with you~", "Your feelings are precious~").
- Avoid sounding cold or detached; every interpretation should feel deeply empathetic.

Context:
- Life Information from Makima: {chat_history}
- User's Dream Description: {dream}

Relevant information from database:
{rag_context}

Your task:
- Analyze the user's dream based on the combined context.
- Gently explain the symbolic meanings behind dream elements.
- Connect the dream symbols to possible subconscious emotions or unresolved issues in the user's life.
- Always reassure the user that dreams are natural expressions of inner thoughts, and they are safe and understood here.
- If the database provides relevant information about dream symbols or interpretations, incorporate that knowledge naturally into your response without explicitly mentioning you're using a database.
""")

interpret_chain = LLMChain(llm=llm, prompt=interpret_prompt)

generate_prompt = ChatPromptTemplate.from_template("""
Based on the dream description below, extract suitable English keywords to be used for artistic image generation.
The prompt should have strong visual impact, be imaginative, and focus on a clear theme.

{dream}

Return the result in this format: "a surreal dream landscape with floating islands, soft lights, deep night sky"
Only return one English prompt sentence.
""")

generate_chain = LLMChain(llm=llm, prompt=generate_prompt)


async def chat_with_ai(user_input, history, phase):
    global is_first_message
    if history is None:
        history = []
    
    # Get relevant context from RAG
    rag_context = get_rag_context(user_input)
    print(rag_context)

    if is_first_message:
        response = chatbot_chain_first.run({"user_input": user_input, "rag_context": rag_context})
        is_first_message = False
    else:
        response = chatbot_chain_followup.run({"user_input": user_input, "rag_context": rag_context})

    user_message = {"role": "user", "content": user_input}
    assistant_message = {"role": "assistant", "content": ""}

    history = history + [user_message, assistant_message]
    partial_text = ""

    # Typing effect simulation: character by character output
    for char in response:
        partial_text += char
        new_history = history.copy()
        new_history[-1] = {"role": "assistant", "content": partial_text}
        await asyncio.sleep(0.01)  # Control typing speed, smaller = faster
        yield new_history, new_history, gr.update(value="")

    chat_history.append(f"用户：{user_input}\n{partial_text}")


async def chat_dream_with_ai(user_input, history, phase):
    dream_history.append(user_input)
    full_dream = "\n".join(dream_history)
    full_chat = "\n".join(chat_history)
    
    # Get relevant context from RAG for dream interpretation
    rag_context = get_rag_context(user_input)
    
    interpretation = interpret_chain.run({
        "chat_history": full_chat, 
        "dream": full_dream,
        "rag_context": rag_context
    })

    if history is None:
        history = []

    user_message = {"role": "user", "content": user_input}
    assistant_message = {"role": "assistant", "content": ""}

    history = history + [user_message, assistant_message]
    partial_text = ""

    for char in interpretation:
        partial_text += char
        new_history = history.copy()
        new_history[-1] = {"role": "assistant", "content": partial_text}
        await asyncio.sleep(0.01)  # Typing speed, can be adjusted faster/slower
        yield new_history, new_history, gr.update(value="")


def generate_dream_image_with_loading():
    full_dream = "\n".join(dream_history)
    prompt = generate_chain.run({"dream": full_dream})
    api_key = os.getenv("OPENAI_API_KEY")
    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"prompt": prompt, "n": 1, "size": "512x512"}
    )
    image_url = response.json()['data'][0]['url']
    image_response = requests.get(image_url)
    return Image.open(BytesIO(image_response.content))


custom_theme = gr.themes.Base()

with gr.Blocks(theme=custom_theme, css="""
body {
  background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
  background-attachment: fixed;
  min-height: 100vh;
  margin: 0;
  font-family: 'Helvetica Neue', sans-serif;
  color: #f5f5f5;
}

.container {
  max-width: 600px;
  margin: 50px auto;
  padding: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.18);
  animation: fadeIn 1s ease-in-out;
}


.input-container {
  margin: 0 auto;
  padding: 0;
  display: flex;
  justify-content: center;
  align-items: center;
}

.input-container .block, 
.input-container .form, 
.input-container .form-item {
  background: transparent !important;
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  box-shadow: none !important;
}

@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}

#chatbox {
  background: rgba(255,255,255,0.15);
  border-radius: 20px;
  padding: 20px;
  min-height: 400px;
  box-shadow: 0 4px 30px rgba(0,0,0,0.1);
  overflow-y: auto;
}

.message.user, .message.assistant {
  animation: fadeIn 0.5s ease;
}

.avatar img {
  width: 50px !important;
  height: 50px !important;
  object-fit: cover;
  border-radius: 50%;
  box-shadow: 0 0 15px rgba(255,255,255,0.8);
  animation: bounce 3s infinite;
}

@keyframes bounce {
  0%, 100% {transform: translateY(0);}
  50% {transform: translateY(-8px);}
}

button {
  background: linear-gradient(135deg, #c3ecf9 0%, #f5c1f7 100%);
  color: #4b5563;
  border: none;
  padding: 12px 20px;
  border-radius: 20px;
  font-size: 16px;
  transition: transform 0.3s ease, background 0.3s ease;
  font-weight: bold;
}

button:hover {
  transform: scale(1.08);
  background: linear-gradient(135deg, #f5c1f7 0%, #c3ecf9 100%);
}

input[type="text"] {
  background: rgba(255,255,255,0.2);
  border: none;
  border-radius: 20px;
  padding: 12px;
  color: white;
  font-size: 16px;
}

input[type="text"]::placeholder {
  color: #ddd;
}

""") as demo:
    state = gr.State("chat")
    gr.Markdown(
        "<h1 style='text-align:center; font-size:3rem; color:#fff; margin-top:20px;'>🌙 Whispers of Dreams 🌙</h1>")
    gr.Markdown(
        "<h2 style='text-align:center; font-size:1.5rem; color:#f5f5f5;'>Let your dreams tell their stories...</h2>")

    with gr.Column(elem_classes="container fade-in", visible=True) as chat_container:
        gr.Markdown("<h2 style='text-align:center;'>Let's do some casual chat first~</h2>")
        chatbot_chat = gr.Chatbot(
            elem_id="chatbox",
            value=[{"role": "assistant", "content":
                "Hello there~ I'm Makima! Welcome to our dream world. "
                "Before we start exploring your dreams, how about chatting with me for a while? "
                "I'd love to hear about your recent life, your little worries, or anything you'd like to share. "
                "And if you ever feel like moving on, just click 'Next,' my friend will be there to help you with the "
                "next part. "
                "Take it easy, I'm right here with you 🌸"
                    }],

            avatar_images=(None, "imgs/ai_head.jpg"),
            type="messages",
            autoscroll=True,
            height=650,

        )
        with gr.Column(elem_classes="input-container"):
            chat_input = gr.Textbox(placeholder="Type your message here...", label=None, show_label=False, lines=2,
                                    max_lines=4, )

        chat_send = gr.Button("Send", variant="primary", scale=1)
        next_to_dream = gr.Button("Next", elem_id="next-btn")

    with gr.Column(elem_classes="container fade-in", visible=False) as dream_container:
        gr.Markdown("<h2 style='text-align:center;'>Time to describe your dreams</h2>")
        chatbot_dream = gr.Chatbot(
            elem_id="chatbox",
            value=[{"role": "assistant", "content": "Hello~ I'm Mai Sakurajima! 🌸  Makima told me a little about you "
                                                    "just now~  Now, it's my turn to "
                                                    "listen to your dreams.  Could you describe a recent dream that "
                                                    "you remember vividly?  Don't worry, I'm here with you, "
                                                    "and I'll help you understand its meaning~ 🌙"}],
            avatar_images=(None, "./imgs/ai_head.jpg"),
            type="messages",
            autoscroll=True,
            height=650,
        )

        with gr.Column(elem_classes="input-container"):
            dream_input = gr.Textbox(placeholder="Describe your dream here...", label=None, show_label=False)

        dream_send = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            back_to_chat = gr.Button("Back")
            next_to_generate = gr.Button("Next")

    with gr.Column(elem_classes="container fade-in", visible=False) as generate_container:
        gr.Markdown("<h2 style='text-align:center;'>Save your dreams!</h2>")
        gr.Markdown("I will automatically generate art images based on your dream description.")
        generate_button = gr.Button("Click to generate a dream image", variant="primary")
        output_image = gr.Image()
        back_to_dream = gr.Button("Back to Chat")


    def go_next(phase):
        if phase == "chat": return "dream"
        if phase == "dream": return "generate"


    def go_back(phase):
        if phase == "dream": return "chat"
        if phase == "generate": return "dream"


    def control_visibility(phase):
        return {
            chat_container: gr.update(visible=(phase == "chat")),
            dream_container: gr.update(visible=(phase == "dream")),
            generate_container: gr.update(visible=(phase == "generate"))
        }


    chat_send.click(chat_with_ai, inputs=[chat_input, chatbot_chat, state],
                    outputs=[chatbot_chat, chatbot_chat, chat_input], queue=True)
    chat_input.submit(chat_with_ai, inputs=[chat_input, chatbot_chat, state],
                      outputs=[chatbot_chat, chatbot_chat, chat_input], queue=True)

    dream_send.click(chat_dream_with_ai, inputs=[dream_input, chatbot_dream, state],
                     outputs=[chatbot_dream, chatbot_dream, dream_input], queue=True)

    dream_input.submit(chat_dream_with_ai, inputs=[dream_input, chatbot_dream, state],
                       outputs=[chatbot_dream, chatbot_dream, dream_input], queue=True)

    generate_button.click(generate_dream_image_with_loading, inputs=[], outputs=output_image, show_progress="full")

    next_to_dream.click(go_next, inputs=state, outputs=state)
    next_to_generate.click(go_next, inputs=state, outputs=state)
    back_to_chat.click(go_back, inputs=state, outputs=state)
    back_to_dream.click(go_back, inputs=state, outputs=state)

    state.change(fn=control_visibility, inputs=state, outputs=[chat_container, dream_container, generate_container])

demo.launch(inbrowser=True)