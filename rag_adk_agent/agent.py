# agent.py
import os
from google.adk.agents import LlmAgent # Using LlmAgent as Agent is an alias
from google.adk.models.lite_llm import LiteLlm
# Assuming rag_tool.py is in the same directory and named correctly
from .rag_tool import query_rag_chain

# Attempt to load OPENAI_API_KEY from .env file if not already in environment
# This helps ensure LiteLLM can find it when ADK runs the agent.
if not os.getenv('OPENAI_API_KEY'):
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key == 'OPENAI_API_KEY':
                        os.environ['OPENAI_API_KEY'] = value.strip('\"').strip("'")
                        break

# Define the ADK agent
root_agent = LlmAgent(
    name="rag_chatbot_agent",
    # Configure to use OpenAI via LiteLlm.
    # User can change 'openai/gpt-3.5-turbo' to other OpenAI models supported by LiteLLM.
    model=LiteLlm(model="openai/gpt-3.5-turbo"),
    description="A RAG chatbot that answers questions based on a provided text file using a specialized tool.",
    instruction="""You are a helpful assistant. Your primary function is to answer questions
    based on the content of a knowledge base. Use the 'query_rag_chain' tool
    to find answers. Provide only the answer from the tool. If the tool returns an error
    or cannot answer, state that you couldn't find the information in the knowledge base.""",
    tools=[query_rag_chain],
    # Default parameters for the tool, if any, can be specified here or when calling the tool
    # Example: default_tool_params={'query_rag_chain': {'training_data_path': 'my_data.txt'}}
)

# To verify OPENAI_API_KEY is loaded (for debugging, can be removed)
# if not os.getenv('OPENAI_API_KEY'):
#    print('AGENT.PY: Warning - OPENAI_API_KEY is not set in the environment after checking .env.')
# else:
#    print(f'AGENT.PY: OPENAI_API_KEY found, starting with prefix: {os.getenv(\'OPENAI_API_KEY\')[:5]}...')
