"""gRPC inference server with real model inference.

Replaces the mock servicer with a real inference pipeline:
- Loads AutoModelForImageClassification + AutoImageProcessor at startup
- Delegates to run_inference() from service.inference.inference_engine
- Maps result scores to prob_ai / prob_human fields
- Returns ERROR status with error_message on inference failure (no crash)

Env vars:
    HF_MODEL_ID      - HuggingFace model ID
                       (default: Ateeqq/ai-vs-human-image-detector)
    GRPC_LOG_LEVEL   - Logging level (default: INFO)
    GRPC_SERVER_HOST - Server host (default: localhost)
    GRPC_SERVER_PORT - Server port (default: 50051)

Usage:
    uv run python -m service.inference_server
"""
import logging
import os
import sys
from concurrent import futures

import grpc
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Add proto/generated to path so gRPC stubs can be imported
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'proto', 'generated'),
)

import inference_pb2  # noqa: E402
import inference_pb2_grpc  # noqa: E402

from inference.inference_engine import run_inference  # noqa: E402

load_dotenv()

LOG_LEVEL = os.getenv('GRPC_LOG_LEVEL', 'INFO')
GRPC_SERVER_HOST = os.getenv('GRPC_SERVER_HOST', '0.0.0.0')
GRPC_SERVER_PORT = int(os.getenv('GRPC_SERVER_PORT', '50051'))
HF_MODEL_ID = os.getenv('HF_MODEL_ID', 'Ateeqq/ai-vs-human-image-detector')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


class AiVsRealClassifierServicer(
    inference_pb2_grpc.AiVsRealClassifierServicer
):
    """gRPC servicer that performs real model inference using HuggingFace."""

    def __init__(self, model=None, processor=None):
        """Load model and processor at startup (injectable for tests).

        Args:
            model: Pre-loaded model. If None, loads from HF_MODEL_ID.
            processor: Pre-loaded processor. If None, loads from HF_MODEL_ID.
        """
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            logger.info('Using injected model and processor.')
        else:
            logger.info(
                'Loading model and processor from HuggingFace: %s',
                HF_MODEL_ID,
            )
            self.processor = AutoImageProcessor.from_pretrained(
                HF_MODEL_ID
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                HF_MODEL_ID
            )
            logger.info('Model and processor loaded successfully.')

    def ClassifyImage(self, request, context):
        logger.info(
            'Received ClassifyImage request: image_id=%s filename=%s',
            request.image_id,
            request.filename,
        )

        # Do not process requests cancelled by the client.
        if not context.is_active():
            logger.warning(
                'Request cancelled before processing: image_id=%s',
                request.image_id,
            )
            context.abort(
                grpc.StatusCode.CANCELLED,
                'Request was cancelled by the client before processing.',
            )
            return inference_pb2.ClassificationResponse()

        try:
            result = run_inference(
                request.image_data, self.model, self.processor
            )
        except Exception as exc:
            # Unexpected error in the inference pipeline: report INTERNAL.
            # Use context.set_code so the client receives a gRPC error.
            logger.exception(
                'Unexpected error during inference for image_id=%s: %s',
                request.image_id,
                exc,
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error interno inesperado: {exc}')
            return inference_pb2.ClassificationResponse(
                image_id=request.image_id,
                status=inference_pb2.ERROR,
                error_message=f'Error interno inesperado: {exc}',
            )

        if result['status'] == 'error':
            error_msg = result['error']['message']
            error_code = result['error'].get('code', 'UNKNOWN')
            logger.warning(
                'Inference error for image_id=%s code=%s: %s',
                request.image_id,
                error_code,
                error_msg,
            )
            # Application-level error (e.g. corrupt image): return an ERROR
            # ClassificationResponse so the client can inspect the message.
            # We do NOT set a gRPC error code here because the server responded
            # successfully at the protocol level; only the image was invalid.
            return inference_pb2.ClassificationResponse(
                image_id=request.image_id,
                status=inference_pb2.ERROR,
                predicted_label='',
                confidence=0.0,
                prob_ai=0.0,
                prob_human=0.0,
                metrics=inference_pb2.PerformanceMetrics(
                    preprocess_time_ms=0,
                    inference_time_ms=0,
                    total_time_ms=0,
                ),
                error_message=error_msg,
            )

        # Map scores dict to proto fields - normalize label keys
        scores = {k.lower(): v for k, v in result['scores'].items()}
        prob_ai = float(scores.get('ai', 0.0))
        # The model may return 'human' or the shorter alias 'hum'
        # for the real/human class; check both keys.
        prob_human_val = (
            scores.get('human')
            if 'human' in scores
            else scores.get('hum', 0.0)
        )
        prob_human = float(prob_human_val)

        # Confidence = score of winning class
        predicted_label = result['label'].lower()
        confidence = float(scores.get(predicted_label, 0.0))

        timing = result['timing']
        metrics = inference_pb2.PerformanceMetrics(
            preprocess_time_ms=int(timing['preprocessing_ms']),
            inference_time_ms=int(timing['inference_ms']),
            total_time_ms=int(timing['total_ms']),
        )

        logger.info(
            'ClassifyImage OK: image_id=%s label=%s confidence=%.4f',
            request.image_id,
            predicted_label,
            confidence,
        )

        return inference_pb2.ClassificationResponse(
            image_id=request.image_id,
            status=inference_pb2.OK,
            predicted_label=predicted_label,
            confidence=confidence,
            prob_ai=prob_ai,
            prob_human=prob_human,
            metrics=metrics,
            error_message='',
        )


def serve(host=None, port=None, model=None, processor=None):
    """Start the gRPC server.

    Args:
        host: Server host (default from env).
        port: Server port (default from env).
        model: Optional pre-loaded model for testing (bypasses HF download).
        processor: Optional pre-loaded processor for testing.
    """
    host = host or GRPC_SERVER_HOST
    port = port or GRPC_SERVER_PORT

    servicer = AiVsRealClassifierServicer(model=model, processor=processor)

    # 50 MB limit to handle large image payloads without RESOURCE_EXHAUSTED.
    _MAX_MSG_BYTES = 50 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", _MAX_MSG_BYTES),
            ("grpc.max_receive_message_length", _MAX_MSG_BYTES),
        ],
    )
    inference_pb2_grpc.add_AiVsRealClassifierServicer_to_server(
        servicer, server
    )
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    server.start()
    logger.info('gRPC server started on port %d', port)
    return server


if __name__ == '__main__':
    server = serve()
    server.wait_for_termination()
