import jinja2
from flask_restful import Resource
from flask import render_template, make_response, request, abort, current_app
from src.utils.SERVICE_MAP import SERVICE_MAP
from src.utils.authentication import authenticate
from src.entity.Payload import Payload


class TemplateResource(Resource):

    @authenticate(endpoint_type='template')
    def get(self, template=None, payload: Payload = None):
        html = None
        if template is None or template == 'index':
            html = render_template(
                "index.html",
            )
        else:
            X_AUTH_TOKEN = request.cookies.get('X-AUTH-TOKEN')
            data = None
            if template in SERVICE_MAP:
                data = SERVICE_MAP[template].get_all()  # TODO: this needs to be based on the user

            try:
                html = render_template(
                    f'{template}.html',
                    data=data
                )
            except jinja2.exceptions.TemplateNotFound:
                current_app.logger.error("template could not be found")
                return abort(404)

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html'

        return response
