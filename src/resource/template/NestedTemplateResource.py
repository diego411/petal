import jinja2.exceptions
from flask import make_response, render_template, abort, current_app
from flask_restful import Resource
from src.utils.authentication import authenticate
from src.entity.Payload import Payload


class NestedTemplateResource(Resource):

    @authenticate(endpoint_type='template')
    def get(self, module: str, template: str, payload: Payload):
        assert payload.resource == 'user', f"Expected payload of resource: 'user' got {payload.resource}"

        try:
            html = render_template(
                f'{module}/{template}.html'
            )
        except jinja2.exceptions.TemplateNotFound:
            current_app.logger.error(f"Template {module}/{template} could not be found!")
            return abort(404)

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html'

        return response
