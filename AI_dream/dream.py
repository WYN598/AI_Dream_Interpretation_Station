import os
import random
import base64
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import gradio as gr
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
import asyncio
from jinja2 import Environment, FileSystemLoader




load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4", streaming=True, temperature=0.9)

# 加载模板目录
env = Environment(loader=FileSystemLoader("prompts"))
template = env.get_template("mai_interpret.txt")

chat_history = []
dream_history = []



# 函数用于读取 prompts
def load_prompt(path: str) -> ChatPromptTemplate:
    with open(path, "r", encoding="utf-8") as f:
        return ChatPromptTemplate.from_template(f.read())


first_prompt = load_prompt("prompts/makima_chat.txt")
chatbot_chain = first_prompt | llm

# interpret_prompt = load_prompt("prompts/mai_interpret.txt")
# interpret_chain = interpret_prompt | llm

interpret_prompt = load_prompt("prompts/mai_interpret.txt")
interpret_short_prompt = load_prompt("prompts/interpret_short.txt")
interpret_chain = interpret_prompt | llm
interpret_short_chain = interpret_short_prompt | llm



generate_prompt = load_prompt("prompts/image_generation.txt")
generate_chain = generate_prompt | llm


async def chat_with_ai(user_input, history, phase):
    if history is None:
        history = []

    # 1. 记录当前用户输入
    user_message = {"role": "user", "content": user_input}
    history.append(user_message)

    # 2. 格式化对话历史为字符串传入 prompt
    formatted_history = "\n".join([
        f"User: {msg['content']}" if msg["role"] == "user" else f"Makima: {msg['content']}"
        for msg in history if msg["content"].strip() != ""
    ])

    # 3. 执行对话调用（invoke）
    response = chatbot_chain.invoke({
        "chat_history": formatted_history,
        "user_input": user_input
    })

    # 4. 处理 AI 响应
    partial_text = ""
    assistant_message = {"role": "assistant", "content": ""}
    history.append(assistant_message)

    for char in response.content:
        partial_text += char
        new_history = history.copy()
        new_history[-1] = {"role": "assistant", "content": partial_text}
        await asyncio.sleep(0.02)
        yield new_history, new_history, gr.update(value="")

    # 5. 存储全局 chat_history 给第二阶段使用
    chat_history.append(f"用户：{user_input}\nMakima：{partial_text}")


# 集成 RAG 检索的 chat_dream_with_ai 函数
async def chat_dream_with_ai(user_input, history, phase):
    # 更新历史
    dream_history.append(user_input)
    full_dream = "\n".join(dream_history)
    full_chat = "\n".join(chat_history)

    # 加载向量数据库
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local("faiss_psych_db", embedding, allow_dangerous_deserialization=True)

    # 检索心理学参考内容
    query = f"{full_dream} {full_chat}"
    results = db.similarity_search(query, k=3)
    retrieved_context = "\n\n".join([doc.page_content for doc in results])

    # 判断梦是否太短
    is_dream_short = len(full_dream.strip().split()) < 20

    # 根据梦境长度选择提示词链
    if is_dream_short:
        response = interpret_short_chain.invoke({})
    else:
        response = interpret_chain.invoke({
            "dream": full_dream,
            "chat_history": full_chat,
            "retrieved_context": retrieved_context
        })

    interpretation = response.content

    # 聊天记录初始化
    if history is None:
        history = []

    user_message = {"role": "user", "content": user_input}
    assistant_message = {"role": "assistant", "content": ""}

    history = history + [user_message, assistant_message]
    partial_text = ""

    # 打字机效果
    for char in interpretation:
        partial_text += char
        new_history = history.copy()
        new_history[-1] = {"role": "assistant", "content": partial_text}
        await asyncio.sleep(0.02)
        yield new_history, new_history, gr.update(value="")


def generate_dream_image_with_style(style):
    full_dream = "\n".join(dream_history)
    prompt = generate_chain.invoke({"dream": full_dream, "style": style}).content
    api_key = os.getenv("OPENAI_API_KEY")

    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "prompt": prompt,
            "n": 1,
            "size": "512x512",
            "response_format": "b64_json"  # ✅ 不返回 URL，直接返回 base64 编码
        }
    )

    image_b64 = response.json()["data"][0]["b64_json"]
    image_bytes = BytesIO(base64.b64decode(image_b64))
    return Image.open(image_bytes)


# 随机图像风格列表
style_list = [
    "Studio Ghibli fantasy style",
    "Surrealism with glowing lights",
    "Soft watercolor illustration",
    "Pixel art dream world",
    "Dark gothic oil painting"
]


# 选取随机图像风格
def generate_random_dream_image():
    style = random.choice(style_list)
    return generate_dream_image_with_style(style)


custom_theme = gr.themes.Base()

with gr.Blocks(theme=custom_theme, css="""
@import url('https://fonts.googleapis.com/css2?family=Caveat&display=swap');

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


#chatbox .avatar-container {
    width: 70px !important;
    height: 70px !important;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, rgba(0,0,0,0) 70%);
    box-shadow: 0 0 10px 2px rgba(173, 216, 230, 0.6),
                0 0 20px 6px rgba(173, 216, 230, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
}

#chatbox .avatar-image {
    width: 95% !important;
    height: 95% !important;
    object-fit: cover !important;
    border-radius: 50%;
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



#chatbox button[aria-label="Clear"] {
    display: none !important;
}



/* 背景渐变 */
.gradio-container {
  background: radial-gradient(ellipse at bottom, #1a1a40 0%, #000014 100%);
  overflow: hidden;
  position: relative;
}


.custom-title {
    text-align:center;
    font-size:2.4rem; 
    background: linear-gradient(90deg, #6c5ce7, #a29bfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Caveat', cursive;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.custom-subtitle{
    text-align:center;
    font-size:1.9rem; 
    background: linear-gradient(90deg, #6c5ce7, #a29bfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Caveat', cursive;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.image-container button {
    background: rgba(0, 0, 0, 0.4);
    border: none;
    padding: none;
    border-radius: 6px;
    margin: 0;
    display: flex;
    color: #4b5563;
    border: none;
    padding: 0;
    border-radius: 20px;
    font-size: 16px;
    transition: none;
}
.image-container button:hover {
    background: none !important;
    color: inherit !important;
    box-shadow: none !important;
    filter: none !important;
    transform: none !important;
    opacity: 1 !important;
    text-decoration: none !important;
    cursor: default !important;

}


""") as demo:
    state = gr.State("chat")

    gr.Markdown(
        "<h1 class='custom-title'>🌙 Whispers of Dreams 🌙</h1>")
    gr.Markdown(
        "<h2 class='custom-title'>Let your dreams tell their stories...</h2>")

    with gr.Column(elem_classes="container fade-in", visible=True) as chat_container:
        gr.Markdown("<h2 class='custom-subtitle'>Let's do some casual chat first</h2>")
        chatbot_chat = gr.Chatbot(

            elem_id="chatbox",
            value=[{"role": "assistant", "content": "Hi, I'm Makima. Welcome to our dream journey!  Before we dive "
                                                    "into dream interpretation, let's have a little chat first.  Tell "
                                                    "me about your recent life — anything that's been on your mind "
                                                    "lately?  If at any point you feel ready to move on, just click "
                                                    "'Next' and my colleague will guide you through the next step.  "
                                                    "Take your time, I'm here to listen."},
                   ],

            avatar_images=(None, "./imgs/ai_head.jpg"),
            type="messages",
            autoscroll=True,
            height=650,
            show_label=False,

        )
        with gr.Column(elem_classes="input-container"):
            chat_input = gr.Textbox(placeholder="Type your message here...", label=None, show_label=False, lines=2,
                                    max_lines=4, )

        chat_send = gr.Button("Send", variant="primary", scale=1)
        next_to_dream = gr.Button("Next", elem_id="next-btn")

    with gr.Column(elem_classes="container fade-in", visible=False) as dream_container:
        gr.Markdown("<h2 class='custom-subtitle'>Time to describe your dreams</h2>")
        chatbot_dream = gr.Chatbot(
            elem_id="chatbox",
            value=[{"role": "assistant", "content": "Hello~ I'm Mai Sakurajima!  Makima told me a little about you "
                                                    "just now~  Now, it's my turn to "
                                                    "listen to your dreams.  Could you describe a recent dream that "
                                                    "you remember vividly?  Don't worry, I'm here with you, "
                                                    "and I'll help you understand its meaning~ 🌙"}],
            avatar_images=(None, "imgs/ai_head2.jpg"),
            type="messages",
            autoscroll=True,
            height=650,
            show_label=False,
        )

        with gr.Column(elem_classes="input-container"):
            dream_input = gr.Textbox(placeholder="Describe your dream here...", label=None, show_label=False)

        dream_send = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            back_to_chat = gr.Button("Back")
            next_to_generate = gr.Button("Next")

    # 存储当前风格的状态变量
    style_state = gr.State(value="Studio Ghibli fantasy style")

    with gr.Column(elem_classes="container fade-in", visible=False) as generate_container:
        gr.Markdown("""
            <h2 style="text-align:center; 
                       font-size:2.2rem; 
                       background: linear-gradient(90deg, #6c5ce7, #a29bfe);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       font-family: 'Caveat', cursive;
                       font-weight: 600;
                       letter-spacing: 1px;
                       margin-bottom: 0.5rem;">
                Thank you for participating in this dream interpretation~
            </h2>
        """)
        gr.Markdown("""
            <h2 style="text-align:center; 
                       font-size:1.8rem; 
                       background: linear-gradient(90deg, #6c5ce7, #a29bfe);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       font-family: 'Caveat', cursive;
                       font-weight: 600;
                       letter-spacing: 1px;
                       margin-top: 0.5rem;">
                Click below to generate your dream painting 🌙
            </h2>
        """)
        # 生成风格选择
        with gr.Column():
            with gr.Row():
                btn_ghibli = gr.Button("Studio Ghibli")
                btn_surreal = gr.Button("Surrealism")
            with gr.Row():
                btn_watercolor = gr.Button("Watercolor")
                btn_pixel = gr.Button("Pixel Art")
            with gr.Row():
                btn_gothic = gr.Button("Gothic")
                btn_random = gr.Button("✨ Random style to generate", variant="secondary")

        output_image = gr.Image(
            show_label=False,
            # show_download_button=False,
            # show_fullscreen_button=False,
        )
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


    # 提交事件绑定
    chat_send.click(chat_with_ai, inputs=[chat_input, chatbot_chat, state],
                    outputs=[chatbot_chat, chatbot_chat, chat_input], queue=True)
    chat_input.submit(chat_with_ai, inputs=[chat_input, chatbot_chat, state],
                      outputs=[chatbot_chat, chatbot_chat, chat_input], queue=True)

    dream_send.click(chat_dream_with_ai, inputs=[dream_input, chatbot_dream, state],
                     outputs=[chatbot_dream, chatbot_dream, dream_input], queue=True)

    dream_input.submit(chat_dream_with_ai, inputs=[dream_input, chatbot_dream, state],
                       outputs=[chatbot_dream, chatbot_dream, dream_input], queue=True)

    # 下一步与返回 按钮点击事件
    next_to_dream.click(go_next, inputs=state, outputs=state)
    next_to_generate.click(go_next, inputs=state, outputs=state)
    back_to_chat.click(go_back, inputs=state, outputs=state)
    back_to_dream.click(go_back, inputs=state, outputs=state)

    state.change(fn=control_visibility, inputs=state, outputs=[chat_container, dream_container, generate_container])

    # 生成图像按钮点击事件
    btn_ghibli.click(generate_dream_image_with_style, inputs=[gr.State("Studio Ghibli fantasy style")],
                     outputs=output_image)
    btn_surreal.click(generate_dream_image_with_style, inputs=[gr.State("Surrealism with glowing lights")],
                      outputs=output_image)
    btn_watercolor.click(generate_dream_image_with_style, inputs=[gr.State("Soft watercolor illustration")],
                         outputs=output_image)
    btn_pixel.click(generate_dream_image_with_style, inputs=[gr.State("Pixel art dream world")], outputs=output_image)
    btn_gothic.click(generate_dream_image_with_style, inputs=[gr.State("Dark gothic oil painting")],
                     outputs=output_image)
    btn_random.click(generate_random_dream_image, outputs=output_image)

# 运行 Gradio 应用
demo.launch(inbrowser=True)
