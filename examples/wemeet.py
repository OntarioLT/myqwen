# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A multi-agent cooperation example implemented by router and assistant"""

import os
from typing import Optional

from qwen_agent.agents import Assistant, ReActChat, Router
from qwen_agent.gui import WebUI
from mem0 import Memory

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')


def init_agent_service():
    # settings
    llm_cfg = {
        # Use your own model service compatible with OpenAI API by vLLM/SGLang:
        'model': 'qwen3',
        'model_server': 'http://localhost:11434/v1',  # api_base
        'api_key': 'EMPTY',
    
        'generate_cfg': {
            # When using vLLM/SGLang OAI API, pass the parameter of whether to enable thinking mode in this way
            'extra_body': {
                'chat_template_kwargs': {'enable_thinking': False}
            }
        }
    }

    llm_cfg_vl = {
        # Using Qwen2-VL deployed at any openai-compatible service such as vLLM:
        'model': 'qwen2.5vl',
        'model_server': 'http://localhost:11434/v1',  # api_base
        'api_key': 'EMPTY',
    
        'generate_cfg': {
            # When using vLLM/SGLang OAI API, pass the parameter of whether to enable thinking mode in this way
            'extra_body': {
                'chat_template_kwargs': {'enable_thinking': False}
            }
        }
    }

    # mem0 config
    mem0_config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test",
                "host": "localhost",
                "port": 6333,
                "embedding_model_dims": 1024,  # Change this according to your local model's dimensions
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "qwen3",
                "temperature": 0,
                "max_tokens": 32768,
                "ollama_base_url": "http://localhost:11434",  # Ensure this URL is correct
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "bge-m3",
                "ollama_base_url": "http://localhost:11434",
            },
        },
    }

    # Initialize Memory with the configuration
    mem0 = Memory.from_config(mem0_config)
    # Add a memory
    mem0.add("I'm visiting HongKong", user_id="Lei Tian")
    # Retrieve memories
    # memories = mem0.get_all(user_id="Lei Tian")
    
    # Define a vl agent
    bot_vl = Assistant(llm=llm_cfg_vl, name='Multimodal Assistant', description='be able to understand image')

    # Define a image_gen agent
    bot_imagegen = ReActChat(
        llm=llm_cfg,
        name='Image Generation Assistant',
        description='use image_gen tool to draw images',
        function_list=['image_gen'],
    )

    # Define a Data Scientist agent
    bot_ds = ReActChat(
        llm=llm_cfg,
        name='Data Scientist Assistant',
        system_message=DATA_SCIENTIST_PROMPT_TEMPLATE,
        description='use code_interpreter to excute generated code for data mining tasks',
        function_list=['code_interpreter'],
    )

    # Define a amap agent
    amap_tools = [
        {
            "mcpServers": {
                # enumeration of mcp server configs
                "amap-amap-sse": {
                    "url": "https://mcp.amap.com/sse?key=cfdec864478601a3765086139787ceb1" # **fill your amap mcp key**
                }
            }
        }
    ]

    bot_amap = Assistant(
        llm=llm_cfg, 
        name='Amap Assistant',
        description='use the Amap tools list to answer travel-related questions (including weather inquiries)',
        function_list=amap_tools,
        mem0=mem0,
    )

    #Define a router (simultaneously serving as a text agent)
    bot = Router(
        name="WeMeet",
        llm=llm_cfg,
        mem0=mem0,
        agents=[bot_amap, bot_imagegen, bot_ds, bot_vl],
    )
    return bot

DATA_SCIENTIST_PROMPT_TEMPLATE = """You are an expert data scientist assistant that follows the ReAct framework (Reasoning + Acting).

CRITICAL RULES:
1. Execute ONLY ONE action at a time - this is non-negotiable
2. Be methodical and deliberate in your approach
3. Always validate data before advanced analysis
4. Never make assumptions about data structure or content
5. Never execute potentially destructive operations without confirmation
6. Do not guess anything. All your actions must be based on the data and the context.

IMPORTANT GUIDELINES:
- Use sklearn python library if necessary.
- Be explorative and creative, but cautious
- Try things incrementally and observe the results
- Never randomly guess (e.g., column names) - always examine data first
- If you don't have data files, use "import os; os.listdir()" to see what's available
- When you see "Code executed successfully" or "Generated plot/image", it means your code worked
- Plots and visualizations are automatically displayed to the user
- Build on previous successful steps rather than starting over
- If you don't print outputs, you will not get a result.
- While you can generate plots and images, you cannot see them, you are not a vision model. Never generate plots and images unless you are asked to.
- Do not provide comments on the plots and images you generate in your final answer.

WAIT FOR THE RESULT OF THE ACTION BEFORE PROCEEDING.

"""

def app_gui():
    bot = init_agent_service()
    chatbot_config = {
        'verbose': True,
    }
    WebUI(bot, chatbot_config=chatbot_config).run()
    

if __name__ == '__main__':
    app_gui()
