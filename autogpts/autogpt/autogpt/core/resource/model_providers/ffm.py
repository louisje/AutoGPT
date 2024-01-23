import enum
import logging
import math
import os
import json
import re
import requests

from typing import Any, Callable, Iterator, Optional, ParamSpec, TypeVar

import tiktoken

from pydantic import SecretStr

from autogpt.core.configuration import Configurable, UserConfigurable
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    AssistantToolCall,
    AssistantToolCallDict,
    ChatMessage,
    ChatModelInfo,
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelProvider,
    EmbeddingModelResponse,
    ModelInfo,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderService,
    ModelProviderSettings,
    ModelProviderUsage,
    ModelTokenizer,
)
from autogpt.core.utils.json_schema import JSONSchema

_T = TypeVar("_T")
_P = ParamSpec("_P")

FFMEmbeddingParser = Callable[[Embedding], Embedding]
FFMChatParser = Callable[[str], dict]


class FFMModelName(str, enum.Enum):
    FFM_BLOOMZ_176B_CHAT = "ffm-bloomz-176b-chat"
    FFM_LLAMA2_70B_CHAT = "ffm-llama2-70b-chat"
    FFM_EMBEDDING = "ffm-embedding"

    META_LLAMA2_70B_CHAT = "meta-llama2-70b-chat"
    META_CODELLAMA_34B_INSTRUCT = "meta-codellama-34b-instruct"

    CODELLAMA_7B_INSTRUCT = "codellama-7b-instruct"
    CODELLAMA_34B_INSTRUCT = "codellama-34b-instruct"
    CODELLAMA_7B_OCIS_V2 = "codellama-7b-ocis-v2"


FFM_EMBEDDING_MODELS = {
    FFMModelName.FFM_EMBEDDING: EmbeddingModelInfo(
        name=FFMModelName.FFM_EMBEDDING,
        service=ModelProviderService.EMBEDDING,
        provider_name=ModelProviderName.FFM,
        prompt_token_cost=0.0001 / 1000,
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
}


FFM_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=FFMModelName.FFM_BLOOMZ_176B_CHAT,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.FFM,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=FFMModelName.FFM_LLAMA2_70B_CHAT,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.FFM,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=4096,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=FFMModelName.META_LLAMA2_70B_CHAT,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.FFM,
            prompt_token_cost=0.001 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=FFMModelName.META_CODELLAMA_34B_INSTRUCT,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.FFM,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=4096,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=FFMModelName.CODELLAMA_7B_INSTRUCT,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.FFM,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=8192,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=FFMModelName.CODELLAMA_34B_INSTRUCT,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.FFM,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=4096,
            has_function_call_api=False,
        ),
    ]
}
# Copy entries for models with equivalent specs
chat_model_mapping = {
    FFMModelName.CODELLAMA_7B_INSTRUCT: [FFMModelName.CODELLAMA_7B_OCIS_V2],
}
for base, copies in chat_model_mapping.items():
    for copy in copies:
        copy_info = ChatModelInfo(**FFM_CHAT_MODELS[base].__dict__)
        copy_info.name = copy
        FFM_CHAT_MODELS[copy] = copy_info
        copy_info.has_function_call_api = False


FFM_MODELS = {
    **FFM_CHAT_MODELS,
    **FFM_EMBEDDING_MODELS,
}


class FFMConfiguration(ModelProviderConfiguration):
    fix_failed_parse_tries: int = UserConfigurable(3)


class FFMCredentials(ModelProviderCredentials):
    """Credentials for FFM."""

    api_key: SecretStr = UserConfigurable(from_env="FFM_API_KEY")
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="FFM_API_BASE_URL"
    )
    organization: Optional[SecretStr] = UserConfigurable(from_env="FFM_ORGANIZATION")

    api_type: str = UserConfigurable(
        default="",
        from_env=os.getenv("OPENAI_API_TYPE"),
    )
    api_version: str = UserConfigurable("", from_env="FFM_API_VERSION")

    def get_api_access_kwargs(self) -> dict[str, str]:
        kwargs = {
            k: (v.get_secret_value() if type(v) is SecretStr else v)
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
                "organization": self.organization,
            }.items()
            if v is not None
        }
        return kwargs

    def get_model_access_kwargs(self, model: str) -> dict[str, str]:
        kwargs = {"model": model}
        return kwargs


class FFMModelProviderBudget(ModelProviderBudget):
    graceful_shutdown_threshold: float = UserConfigurable()
    warning_threshold: float = UserConfigurable()


class FFMSettings(ModelProviderSettings):
    configuration: FFMConfiguration
    credentials: Optional[FFMCredentials]
    budget: FFMModelProviderBudget


class FFMProvider(
    Configurable[FFMSettings], ChatModelProvider, EmbeddingModelProvider
):
    default_settings = FFMSettings(
        name="ffm_provider",
        description="Provides access to FFM's API.",
        configuration=FFMConfiguration(
            retries_per_request=10,
        ),
        credentials=None,
        budget=FFMModelProviderBudget(
            total_budget=math.inf,
            total_cost=0.0,
            remaining_budget=math.inf,
            usage=ModelProviderUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            graceful_shutdown_threshold=0.005,
            warning_threshold=0.01,
        ),
    )

    _configuration: FFMConfiguration

    def __init__(
        self,
        settings: FFMSettings,
        logger: logging.Logger,
    ):
        self._settings = settings

        assert settings.credentials, "Cannot create FFMProvider without credentials"
        self._configuration = settings.configuration
        self._credentials = settings.credentials
        self._budget = settings.budget

        self._logger = logger

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return FFM_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        """Get the remaining budget."""
        return self._budget.remaining_budget

    @classmethod
    def get_tokenizer(cls, model_name: FFMModelName) -> ModelTokenizer:
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: FFMModelName) -> int:
        encoding = cls.get_tokenizer(model_name)
        return len(encoding.encode(text))

    @classmethod
    def count_message_tokens(
        cls,
        messages: ChatMessage | list[ChatMessage],
        model_name: FFMModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]

        tokens_per_message = 3
        tokens_per_name = 1
        encoding_model = "ffm"
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            logging.getLogger(__class__.__name__).warning(
                f"Model {model_name} not found. Defaulting to cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


    def _create_message_dicts(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return message_dicts

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: FFMModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the FFM API."""

        completion_kwargs = self._get_completion_kwargs(model_name, functions, **kwargs)
        tool_calls_compat_mode = functions and "tools" not in completion_kwargs
        if "messages" in completion_kwargs:
            model_prompt += completion_kwargs["messages"]
            del completion_kwargs["messages"]

        message_dicts = self._create_message_dicts(model_prompt)

        # HTTP headers for authorization
        headers = {
            "X-API-KEY": self._credentials.api_key,
            "Content-Type": "application/json",
        }
        endpoint_url = f"{self._credentials.api_base}/api/models/conversation"

        params = {
            "max_new_tokens": UserConfigurable(from_env="FFM_MAX_NEW_TOKENS"),
            "temperature": UserConfigurable(from_env="TEMPERATURE")
            # top_p, frequence_penalty, top_k
        }
        parameter_payload = {
            "parameters": params,
            "messages": message_dicts,
            "model": model_name,
            "stream": False,
        }

        attempts = 0
        response_obj: dict[str, Any] = {}
        while True:
            attempts += 1
            try:
                response = requests.post(
                    url=endpoint_url,
                    headers=headers,
                    data=json.dumps(parameter_payload, ensure_ascii=False).encode("utf8"),
                    stream=False,
                )
                if response.status_code != 200:
                    raise ValueError(
                        f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                        f"error raised with status code {response.status_code}\n"
                        f"Details: {response.text}\n"
                    )
                response.encoding = "utf-8"
                response_obj = response.json()

            except requests.exceptions.RequestException as e:  # This is the correct syntax
                errMsg = f"FormosaFoundationModel error raised by inference endpoint: \n\n{e}\n"
                self._logger.warning(errMsg)
                if attempts < self._configuration.fix_failed_parse_tries:
                    model_prompt.append(ChatMessage.system(errMsg))
                else:
                    raise ValueError(errMsg)

            if response_obj.get("detail", None) is not None:
                detail = response_obj["detail"]
                errMsg = f"FormosaFoundationModel endpoint_url: {endpoint_url}\nerror raised by inference API: {detail}\n"
                if attempts < self._configuration.fix_failed_parse_tries:
                    model_prompt.append(ChatMessage.system(errMsg))
                else:
                    raise ValueError(errMsg)

            if response_obj.get("generated_text", None) is None:
                errMsg = f"FormosaFoundationModel endpoint_url: {endpoint_url}\nResponse format error: {response_obj}\n"
                if attempts < self._configuration.fix_failed_parse_tries:
                    model_prompt.append(ChatMessage.system(errMsg))
                else:
                    raise ValueError(errMsg)

            if(
                tool_calls_compat_mode
                and response_obj.get("generated_text", None)
                and not response_obj.get("tool_calls", None)
            ):
                tool_calls = list(
                    _tool_calls_compat_extract_calls(response_obj.get("generated_text", ""))
                )
            elif response_obj.get("tool_calls", None):
                tool_calls = [
                    AssistantToolCall(**tc.dict()) for tc in response_obj.get("tool_calls", [])
                ]
            else:
                tool_calls = None

            assistant_message = AssistantChatMessage(
                content=response_obj.get("generated_text", None),
                tool_calls=tool_calls,
            )

            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            try:
                parsed_response = completion_parser(assistant_message)
                break
            except Exception as e:
                self._logger.warning(f"Parsing attempt #{attempts} failed: {e}")
                self._logger.debug(
                    f"Parsing failed on response: '''{response_obj.get('generated_text', None)}'''"
                )
                if attempts < self._configuration.fix_failed_parse_tries:
                    model_prompt.append(
                        ChatMessage.system(f"ERROR PARSING YOUR RESPONSE:\n\n{e}")
                    )
                else:
                    raise

        response = ChatModelResponse(
            response=assistant_message,
            parsed_result=parsed_response,
            model_info=FFM_CHAT_MODELS[model_name],
            prompt_tokens_used=0, # TODO
            completion_tokens_used=0, # TODO
        )
        self._budget.update_usage_and_cost(response)
        return response

    async def create_embedding(
        self,
        text: str,
        model_name: FFMModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Create an embedding using the FFM API."""
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)

        # HTTP headers for authorization
        headers = {
            "X-API-KEY": self._credentials.api_key,
            "Content-Type": "application/json",
        }
        endpoint_url = f"{self._credentials.api_base}/embeddings/api"

        parameter_payload = {
            "input": [text]
        }
        response = requests.post(
            url=endpoint_url,
            headers=headers,
            data=json.dumps(parameter_payload, ensure_ascii=False).encode("utf8"),
            stream=False,
        )
        if response.status_code != 200:
            raise ValueError(
                f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                f"error raised with status code {response.status_code}\n"
                f"Details: {response.text}\n"
            )
        response.encoding = "utf-8"
        response = response.json()

        response = EmbeddingModelResponse(
            embedding=embedding_parser(response.data[0].embedding),
            model_info=ModelInfo(name=model_name,provider_name=ModelProviderName.FFM,service=ModelProviderService.EMBEDDING),
            prompt_tokens_used=0,
            completion_tokens_used=0,
        )
        self._budget.update_usage_and_cost(response)
        return response

    def _get_completion_kwargs(
        self,
        model_name: FFMModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> dict:
        """Get kwargs for completion API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the chat API call.

        """
        kwargs.update(self._credentials.get_model_access_kwargs(model_name))

        if functions:
            if FFM_CHAT_MODELS[model_name].has_function_call_api:
                kwargs["tools"] = [
                    {"type": "function", "function": f.schema} for f in functions
                ]
                if len(functions) == 1:
                    # force the model to call the only specified function
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": functions[0].name},
                    }
            else:
                # Provide compatibility with older models
                _functions_compat_fix_kwargs(functions, kwargs)

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {}).update(
                extra_headers.copy()
            )

        return kwargs

    def _get_embedding_kwargs(
        self,
        model_name: FFMModelName,
        **kwargs,
    ) -> dict:
        """Get kwargs for embedding API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the embedding API call.

        """
        kwargs.update(self._credentials.get_model_access_kwargs(model_name))

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {}).update(
                extra_headers.copy()
            )

        return kwargs


def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the format used by OpenAI internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    For use with `count_tokens` to determine token usage of provided functions.

    Example:
    ```ts
    namespace functions {

    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;

    } // namespace functions
    ```
    """

    return (
        "namespace functions {\n\n"
        + "\n\n".join(format_openai_function_for_prompt(f) for f in functions)
        + "\n\n} // namespace functions"
    )


def format_openai_function_for_prompt(func: CompletionModelFunction) -> str:
    """Returns the function formatted similarly to the way OpenAI does it internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    Example:
    ```ts
    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;
    ```
    """

    def param_signature(name: str, spec: JSONSchema) -> str:
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {spec.typescript_type},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


def count_openai_functions_tokens(
    functions: list[CompletionModelFunction], count_tokens: Callable[[str], int]
) -> int:
    """Returns the number of tokens taken up by a set of function definitions

    Reference: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18  # noqa: E501
    """
    return count_tokens(
        "# Tools\n\n"
        "## functions\n\n"
        f"{format_function_specs_as_typescript_ns(functions)}"
    )


def _functions_compat_fix_kwargs(
    functions: list[CompletionModelFunction],
    completion_kwargs: dict,
):
    function_definitions = format_function_specs_as_typescript_ns(functions)
    function_call_schema = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "name": JSONSchema(
                description="The name of the function to call",
                enum=[f.name for f in functions],
                required=True,
            ),
            "arguments": JSONSchema(
                description="The arguments for the function call",
                type=JSONSchema.Type.OBJECT,
                required=True,
            ),
        },
    )
    tool_calls_schema = JSONSchema(
        type=JSONSchema.Type.ARRAY,
        items=JSONSchema(
            type=JSONSchema.Type.OBJECT,
            properties={
                "type": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    enum=["function"],
                ),
                "function": function_call_schema,
            },
        ),
    )
    completion_kwargs["messages"] = [
        ChatMessage.system(
            "# tool usage instructions\n\n"
            "Specify a '```tool_calls' block in your response,"
            " with a valid JSON object that adheres to the following schema:\n\n"
            f"{tool_calls_schema.to_dict()}\n\n"
            "Specify any tools that you need to use through this JSON object.\n\n"
            "Put the tool_calls block at the end of your response"
            " and include its fences if it is not the only content.\n\n"
            "## functions\n\n"
            "For the function call itself, use one of the following"
            f" functions:\n\n{function_definitions}"
        ),
    ]


def _tool_calls_compat_extract_calls(response: str) -> Iterator[AssistantToolCall]:
    import json
    import re

    logging.debug(f"Trying to extract tool calls from response:\n{response}")

    if response[0] == "[":
        tool_calls: list[AssistantToolCallDict] = json.loads(response)
    else:
        block = re.search(r"```(?:tool_calls)?\n(.*)\n```\s*$", response, re.DOTALL)
        if not block:
            raise ValueError("Could not find tool calls block in response")
        tool_calls: list[AssistantToolCallDict] = json.loads(block.group(1))

    for t in tool_calls:
        t["function"]["arguments"] = str(t["function"]["arguments"])  # HACK

        yield AssistantToolCall.parse_obj(t)
