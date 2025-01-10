from flask import render_template, make_response
from flask_restful import Resource
from src.service import recording_service


class EntityTemplateResource(Resource):

    def get(self, template: str, entity_id: str):
        html = None
        if template == 'recording':
            recording = recording_service.find_by_id(entity_id)
            if recording is None:
                return render_template(
                    'empty_recording.html',
                    id=entity_id  # TODO: fix naming
                )

            html = render_template(
                'recording.html',
                recording=recording_service.to_dto(recording),
            )

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html'

        return response
