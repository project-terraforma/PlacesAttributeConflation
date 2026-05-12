import unittest

from places_attr_conflation.lab_config import LabConfig, ProjectPolicy, ProviderConfig, RoutingPolicy
from places_attr_conflation.lab_protocol import GenerateRequest, LabPolicyInput
from places_attr_conflation.lab_router import route_generate_request


class LabRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LabConfig(
            providers=(
                ProviderConfig(name="local-dev", kind="local", model="qwen-coder"),
                ProviderConfig(name="hosted-backup", kind="hosted", model="gpt-5.5", api_key_env="LAB_API_KEY"),
            ),
            project_policy=ProjectPolicy(mode="hybrid", allow_hosted_fallback=True),
            routing_policy=RoutingPolicy(long_context_chars=25, high_reasoning_tasks=("research_synthesis",)),
        )

    def test_default_route_prefers_local(self):
        decision = route_generate_request(
            self.config,
            GenerateRequest(task="generate", prompt="short", context="", policy=LabPolicyInput(allow_hosted_fallback=True)),
        )
        self.assertEqual(decision.provider_name, "local-dev")
        self.assertFalse(decision.used_hosted_fallback)

    def test_long_context_uses_hosted_when_allowed(self):
        decision = route_generate_request(
            self.config,
            GenerateRequest(
                task="generate",
                prompt="x" * 20,
                context="y" * 20,
                policy=LabPolicyInput(allow_hosted_fallback=True),
            ),
        )
        self.assertEqual(decision.provider_name, "hosted-backup")
        self.assertTrue(decision.used_hosted_fallback)

    def test_high_reasoning_stays_local_when_request_disallows_fallback(self):
        decision = route_generate_request(
            self.config,
            GenerateRequest(
                task="research_synthesis",
                prompt="outline the paper",
                context="",
                policy=LabPolicyInput(allow_hosted_fallback=False),
            ),
        )
        self.assertEqual(decision.provider_name, "local-dev")
        self.assertFalse(decision.used_hosted_fallback)


if __name__ == "__main__":
    unittest.main()
