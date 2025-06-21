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
    # Add an inital memory
    mem0.add("This is local persistent memory for agent wemeet", user_id="wemeet", metadata={"category": 'description'})

    # Define a vl agent
    bot_vl = Assistant(llm=llm_cfg_vl, name='多模态助手', mem0=mem0, description='可以理解图像内容。')

    # Define a image_gen agent
    bot_imagegen = ReActChat(
        llm=llm_cfg,
        name='工具助手',
        description='可以使用画图工具和运行代码来解决问题',
        function_list=['image_gen', 'code_interpreter'],
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
        name='高德地图助手',
        description='可以使用高德地图工具列表回答出行相关问题（可查询天气）',
        function_list=amap_tools,
        mem0=mem0,
    )

    #Define a router (simultaneously serving as a text agent)
    bot = Router(
        name="WeMeet",
        llm=llm_cfg,
        mem0=mem0,
        agents=[bot_amap,  bot_vl, bot_imagegen],
    )
    return bot


def app_gui():
    bot = init_agent_service()
    chatbot_config = {
        'verbose': True,
    }
    WebUI(bot, chatbot_config=chatbot_config).run()
    

if __name__ == '__main__':
    app_gui()
