# Copyright 2026 Sentient Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import json
import os

from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource(attributes={"service.name": "dag-orchestrator"})
provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter()
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

DSPyInstrumentor().instrument()
LiteLLMInstrumentor().instrument()


def trace_and_capture(func):
    def prepare_data(obj):
        if obj is None:
            return None
        if isinstance(obj, (list, tuple)):
            return [prepare_data(i) for i in obj]

        for method in ("model_dump_json", "json"):
            m = getattr(obj, method, None)
            if callable(m):
                try:
                    return json.loads(m())
                except:
                    continue

        for method in ("model_dump", "toDict", "dict", "as_dict"):
            m = getattr(obj, method, None)
            if callable(m):
                try:
                    return m()
                except:
                    continue

        if hasattr(obj, "__dict__"):
            try:
                return vars(obj)
            except:
                pass

        return obj

    def safe_serialize(data):
        try:
            return json.dumps(data, default=str)
        except:
            return str(data)

    def validate_args(args, func_name):
        if len(args) > 1:
            raise ValueError(
                f"Function '{func_name}' supports ONLY keyword arguments. Too many positional args found: {args}"
            )

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            validate_args(args, func.__name__)

            with tracer.start_as_current_span(func.__name__) as span:
                clean_kwargs = {k: prepare_data(v) for k, v in kwargs.items()}
                span.set_attribute("input.value", safe_serialize(clean_kwargs))

                result = await func(*args, **kwargs)

                clean_result = prepare_data(result)
                span.set_attribute("output.value", safe_serialize(clean_result))
                return result

        return wrapper

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validate_args(args, func.__name__)

            with tracer.start_as_current_span(func.__name__) as span:
                clean_kwargs = {k: prepare_data(v) for k, v in kwargs.items()}
                span.set_attribute("input.value", safe_serialize(clean_kwargs))

                result = func(*args, **kwargs)

                clean_result = prepare_data(result)
                span.set_attribute("output.value", safe_serialize(clean_result))
                return result

        return wrapper
