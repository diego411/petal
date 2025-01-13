from flask import render_template, make_response, abort
from flask_restful import Resource
from src.service import recording_service
from src.utils.SERVICE_MAP import SERVICE_MAP


def get_data(template: str, entity_id: str):
    if template == 'recording':
        recording = recording_service.find_by_id(entity_id)
        return {
            'recording': recording_service.to_dto(recording)
        }


class EntityTemplateResource(Resource):

    def get(self, template: str, entity_id: str):
        if template not in SERVICE_MAP:
            abort(404)

        entity = SERVICE_MAP[template].find_by_id(entity_id)

        if entity is None:
            abort(404)

        html = render_template(
            f'{template}.html',
            data=get_data(template, entity_id)
        )

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html'

        return response
