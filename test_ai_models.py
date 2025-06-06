import unittest
import os
import importlib # Added for reloading rag_chatbot
from unittest.mock import patch, MagicMock

# Import classes to be tested
from ai_models import AIModel, OpenAIModel, JulesModel, MCPModel

# Imports for type checking and mocking internal components if necessary
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

# Import the module containing the main logic for model selection
# We need to be careful here if rag_chatbot.py has side effects on import
# or relies heavily on environment variables being set at module level.
# For now, we'll import it directly.
import rag_chatbot

class TestOpenAIModel(unittest.TestCase):
    def test_initialization_and_getters(self):
        """Test OpenAIModel initialization and its getter methods."""
        # Provide a dummy API key for initialization
        dummy_api_key = "sk-testkey123"
        model = OpenAIModel(api_key=dummy_api_key)

        # Check if get_embeddings() returns an instance of OpenAIEmbeddings
        self.assertIsInstance(model.get_embeddings(), OpenAIEmbeddings,
                              "get_embeddings() should return an OpenAIEmbeddings instance.")

        # Check if generate_response() returns an instance of OpenAI (the LLM)
        self.assertIsInstance(model.generate_response(), OpenAI,
                              "generate_response() should return an OpenAI LLM instance.")

class TestJulesModel(unittest.TestCase):
    def test_placeholder_methods(self):
        """Test JulesModel placeholder methods raise NotImplementedError."""
        dummy_api_key = "jules-testkey123" # Dummy key, as it's a placeholder
        model = JulesModel(api_key=dummy_api_key)

        with self.assertRaises(NotImplementedError, msg="JulesModel.get_embeddings should raise NotImplementedError"):
            model.get_embeddings()

        with self.assertRaises(NotImplementedError, msg="JulesModel.generate_response should raise NotImplementedError"):
            model.generate_response()

class TestMCPModel(unittest.TestCase):
    def test_placeholder_methods(self):
        """Test MCPModel placeholder methods raise NotImplementedError."""
        dummy_api_key = "mcp-testkey123" # Dummy key, as it's a placeholder
        model = MCPModel(api_key=dummy_api_key)

        with self.assertRaises(NotImplementedError, msg="MCPModel.get_embeddings should raise NotImplementedError"):
            model.get_embeddings()

        with self.assertRaises(NotImplementedError, msg="MCPModel.generate_response should raise NotImplementedError"):
            model.generate_response()


class TestChatbotModelSelection(unittest.TestCase):

    def _run_main_logic_for_test(self):
        """
        Helper to encapsulate the call to rag_chatbot.main() and expected error handling.
        This avoids repeating the try-except block in each test.
        """
        try:
            rag_chatbot.main()
        except (NotImplementedError, RuntimeError, ValueError, SystemExit) as e:
            # Catch errors from placeholder models, missing keys if not mocked, or SystemExit if main exits early.
            print(f"Caught expected exception/exit in test: {e}")
            pass

    @patch('rag_chatbot.load_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.split_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.setup_rag_pipeline')
    @patch('ai_models.OpenAIModel')
    def test_select_openai_model(self, MockOpenAIModelConstructor, mock_setup_rag, mock_split, mock_load):
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'AI_MODEL_PROVIDER': 'OPENAI',
                'OPENAI_API_KEY': 'fake_openai_key'
            }.get(key, default)
            with patch.dict(rag_chatbot.os.environ, {'OPENAI_API_KEY': 'fake_openai_key'}): # This helps for os.environ access within main
                rag_chatbot.OPENAI_API_KEY = 'fake_openai_key' # Set module level variable directly
                importlib.reload(rag_chatbot) # Reload to apply mock to module-level getenv
                self._run_main_logic_for_test()
                MockOpenAIModelConstructor.assert_called_once_with(api_key='fake_openai_key')

    @patch('rag_chatbot.load_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.split_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.setup_rag_pipeline')
    @patch('ai_models.JulesModel')
    def test_select_jules_model(self, MockJulesModelConstructor, mock_setup_rag, mock_split, mock_load):
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'AI_MODEL_PROVIDER': 'JULES',
                'JULES_API_KEY': 'fake_jules_key',
                'OPENAI_API_KEY': 'fake_openai_key'
            }.get(key, default)
            with patch.dict(rag_chatbot.os.environ, {'OPENAI_API_KEY': 'fake_openai_key'}):
                rag_chatbot.OPENAI_API_KEY = 'fake_openai_key'
                importlib.reload(rag_chatbot)
                self._run_main_logic_for_test()
                MockJulesModelConstructor.assert_called_once_with(api_key='fake_jules_key')

    @patch('rag_chatbot.load_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.split_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.setup_rag_pipeline')
    @patch('ai_models.MCPModel')
    def test_select_mcp_model(self, MockMCPModelConstructor, mock_setup_rag, mock_split, mock_load):
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'AI_MODEL_PROVIDER': 'MCP',
                'MCP_API_KEY': 'fake_mcp_key',
                'OPENAI_API_KEY': 'fake_openai_key'
            }.get(key, default)
            with patch.dict(rag_chatbot.os.environ, {'OPENAI_API_KEY': 'fake_openai_key'}):
                rag_chatbot.OPENAI_API_KEY = 'fake_openai_key'
                importlib.reload(rag_chatbot)
                self._run_main_logic_for_test()
                MockMCPModelConstructor.assert_called_once_with(api_key='fake_mcp_key')

    @patch('rag_chatbot.load_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.split_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.setup_rag_pipeline')
    @patch('ai_models.OpenAIModel')
    def test_select_default_model(self, MockOpenAIModelConstructor, mock_setup_rag, mock_split, mock_load):
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'OPENAI_API_KEY': 'fake_openai_key'
            }.get(key, default)
            with patch.dict(rag_chatbot.os.environ, {'OPENAI_API_KEY': 'fake_openai_key'}):
                rag_chatbot.OPENAI_API_KEY = 'fake_openai_key'
                importlib.reload(rag_chatbot)
                self._run_main_logic_for_test()
                MockOpenAIModelConstructor.assert_called_once_with(api_key='fake_openai_key')

    @patch('rag_chatbot.load_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.split_documents', return_value=[MagicMock()])
    @patch('rag_chatbot.setup_rag_pipeline')
    @patch('builtins.print')
    @patch('ai_models.OpenAIModel')
    @patch('ai_models.JulesModel')
    @patch('ai_models.MCPModel')
    def test_select_unknown_model_provider(self, MockMCPModel, MockJulesModel, MockOpenAIModel, mock_print, mock_setup_rag, mock_split, mock_load):
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default=None: {
                'AI_MODEL_PROVIDER': 'UNKNOWN_PROVIDER',
                'OPENAI_API_KEY': 'fake_openai_key'
            }.get(key, default)
            with patch.dict(rag_chatbot.os.environ, {'OPENAI_API_KEY': 'fake_openai_key'}):
                rag_chatbot.OPENAI_API_KEY = 'fake_openai_key'
                importlib.reload(rag_chatbot)
                self._run_main_logic_for_test()

                MockOpenAIModel.assert_not_called()
                MockJulesModel.assert_not_called()
                MockMCPModel.assert_not_called()
                # Check if the specific error print for unknown provider was called
                mock_print.assert_any_call("Error: Unknown AI_MODEL_PROVIDER 'UNKNOWN_PROVIDER'. Please use 'OPENAI', 'JULES', or 'MCP'.")

if __name__ == '__main__':
    unittest.main()
