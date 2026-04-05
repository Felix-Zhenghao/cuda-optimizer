# Debugging

Describes how to enable cuDNN frontend API logging for debugging. Logging is controlled via environment variables CUDNN_FRONTEND_LOG_INFO and CUDNN_FRONTEND_LOG_FILE (output to stdout, stderr, or a file), or programmatically via cudnn_frontend::isLoggingEnabled() and cudnn_frontend::getStream() API calls.
