from src.service import recording_service, script_service, live_emotion_service, experiment_service

SERVICE_MAP = {
    'live-emotion': live_emotion_service,
    'recording': recording_service,
    'recordings': recording_service,
    'scripts': script_service,
    'experiments': experiment_service,
    'experiment': experiment_service
}
