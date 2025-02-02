from flask import render_template, make_response, abort
from flask_restful import Resource
from src.service import recording_service, experiment_service
from src.utils.SERVICE_MAP import SERVICE_MAP
from src.utils.authentication import authenticate
from src.entity.Payload import Payload
from src.entity.Recording import Recording
from src.entity.exception.UnauthorizedException import UnauthorizedException
from src.entity.Experiment import Experiment


def get_data(entity, payload: Payload):
    if isinstance(entity, Recording):
        return {
            'recording': recording_service.to_dto(entity)
        }
    if isinstance(entity, Experiment):
        return {
            'experiment': experiment_service.to_dto(entity),
            'recordings': recording_service.get_all_without_experiment(payload.id)
        }


class EntityTemplateResource(Resource):

    @authenticate(endpoint_type='template')
    def get(self, template: str, entity_id: str, payload: Payload):
        if template not in SERVICE_MAP:
            abort(404)

        entity = SERVICE_MAP[template].find_by_id(entity_id)

        if entity is None:
            abort(404)

        if entity.user is not None and entity.user != payload.id:
            raise UnauthorizedException('template', "403 - You are not authorized to access this entity!")

        html = render_template(
            f'{template}.html',
            data=get_data(entity, payload)
        )

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html'

        return response
