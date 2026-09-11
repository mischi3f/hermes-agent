from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def test_credential_pool_rotation_updates_compression_authority_without_changing_model():
    agent = object.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent.api_key = "old-authority"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.provider = "openai-codex"
    agent.model = "gpt-5.5"
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": agent.base_url}
    agent._reapply_route_client_config = MagicMock()
    agent._replace_primary_openai_client = MagicMock()
    agent.context_compressor = SimpleCompressor(
        model="gpt-5.5", api_key="old-authority", base_url=agent.base_url
    )
    entry = MagicMock()
    entry.id = "cred-b"
    entry.runtime_api_key = "rotated-authority"
    entry.runtime_base_url = agent.base_url

    with patch("hermes_cli.route_identity.normalize_route_base_url", side_effect=lambda value: value):
        agent._swap_credential(entry)

    assert agent.api_key == "rotated-authority"
    assert agent.context_compressor.api_key == "rotated-authority"
    assert agent.context_compressor.model == "gpt-5.5"
    assert agent.context_compressor.base_url == agent.base_url


class SimpleCompressor:
    def __init__(self, *, model, api_key, base_url):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
