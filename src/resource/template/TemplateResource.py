from flask_restful import Resource
from flask import render_template, make_response
import os
from src.service import recording_service
from src.entity.RecordingState import RecordingState
from pathlib import Path
import json

current_emotion = "neutral"


class TemplateResource(Resource):

    def get(self, template=None):
        html = None
        if template is None:
            html = render_template(
                "index.html",
            )

        if template == 'live-emotion':
            html = render_template(
                "live_emotion.html",
                initial_emotion=current_emotion,
                initial_image_src=os.path.join('static', f"{current_emotion}.svg")
            )

        if template == 'audio-classification':
            html = render_template(
                "audio_classification.html"
            )

        if template == 'recordings':
            recordings = recording_service.find_by_state(RecordingState.REGISTERED) + recording_service.find_by_state(
                RecordingState.RUNNING)

            html = render_template(
                "recordings.html",
                recordings=recording_service.to_dtos(recordings)
            )

        if template == 'label-recordings':
            html = render_template(
                'label.html'
            )

        if template == 'record-and-label':
            recordings = recording_service.find_by_state(RecordingState.REGISTERED) + recording_service.find_by_state(
                RecordingState.RUNNING)

            html = render_template(
                'record_and_label.html',
                recordings=recording_service.to_dtos(recordings)
            )

        if template == 'scripts':
            scripts = []
            path = Path('scripts')
            for item in path.iterdir():
                if item.is_dir():
                    script = {"name": item.name, "versions": []}
                    versions = []
                    for version in item.iterdir():
                        if not version.is_dir() and version.suffix == '.npy':
                            content = ''
                            with version.open('r') as file:
                                content = file.read()  # .replace('"', '\\"').replace("'", "\\'")
                            versions.append({"identifier": version.name.split('.')[0], "content": content})

                    sorted(versions, key=lambda element: element['identifier'])
                    versions[-1]['identifier'] += " (latest)"
                    script['versions'] = versions
                    scripts.append(script)

            html = render_template(
                'scripts.html',
                scripts=scripts,
                parsed_scripts=json.dumps(scripts)
            )

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html'

        return response