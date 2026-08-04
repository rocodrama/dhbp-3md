import unittest

from threemd.serving import RemoteVLLMServer


class RemoteVLLMServerTests(unittest.TestCase):
    def test_start_command_uses_ssh_and_remote_vllm_api_server(self) -> None:
        server = RemoteVLLMServer(
            host="gpu-box",
            model="google/medgemma-1.5-4b-it",
            port=8000,
            gpu_devices="0",
        )

        command = server.start_command()

        self.assertEqual(command[0:2], ["ssh", "gpu-box"])
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command[2])
        self.assertTrue(command[2].startswith("CUDA_VISIBLE_DEVICES=0 nohup "))
        self.assertIn("vllm serve google/medgemma-1.5-4b-it", command[2])
        self.assertIn("--port 8000", command[2])

    def test_health_url_points_to_openai_models_endpoint(self) -> None:
        server = RemoteVLLMServer(host="gpu-box", model="m", port=9000)

        self.assertEqual(
            server.health_url("http://127.0.0.1:9000/v1"),
            "http://127.0.0.1:9000/v1/models",
        )

    def test_vllm_bin_can_include_conda_run_prefix(self) -> None:
        server = RemoteVLLMServer(
            host="gpu-box",
            model="google/medgemma-1.5-4b-it",
            vllm_bin="conda run -n vllm vllm",
        )

        command = server.start_command()[2]

        self.assertIn("conda run -n vllm vllm serve google/medgemma-1.5-4b-it", command)


if __name__ == "__main__":
    unittest.main()
