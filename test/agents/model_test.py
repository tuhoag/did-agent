from agents import ModelFactory


def test_model_registration_with_ollama():
    factory = ModelFactory()
    model_name = "qwen2.5-coder:0.5b"
    backend = "ollama"
    factory.register_model(model_name, backend)
    model = factory.get_model(model_name)
    assert model is not None
    assert model.name == model_name
    assert model.backend == backend

def test_model_registration_with_vllm():
    factory = ModelFactory()
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    backend = "vllm"
    factory.register_model(model_name, backend)
    model = factory.get_model(model_name)
    assert model is not None
    assert model.name == model_name
    assert model.backend == backend

def test_model_inference_with_ollama():
    factory = ModelFactory()
    model_name = "qwen2.5-coder:0.5b"
    backend = "ollama"
    factory.register_model(model_name, backend)
    model = factory.get_model(model_name)
    response = model.invoke([{"role": "user", "content": "Hello, how are you?"}])
    assert response is not None
