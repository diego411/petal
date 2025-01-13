import jinja2
from flask_restful import Resource
from flask import render_template, make_response, request, abort
from src.utils.SERVICE_MAP import SERVICE_MAP


class TemplateResource(Resource):

    def get(self, template=None):
        html = None
        if template is None:
            html = render_template(
                "index.html",
            )
        else:
            X_AUTH_TOKEN = request.cookies.get('X-AUTH-TOKEN')
            data = None
            if template in SERVICE_MAP:
                data = SERVICE_MAP[template].get_all()

            try:
                html = render_template(
                    f'{template}.html',
                    data=data
                )
            except jinja2.exceptions.TemplateNotFound:
                return abort(404)

        response = make_response(html)
        response.headers['Content-Type'] = 'text/html'

        return response
