from flask import render_template, current_app, redirect, url_for
from flask_restful import Api
import werkzeug


class PlantApi(Api):

    def handle_error(self, e):
        current_app.logger.error(e)
        if isinstance(e, werkzeug.exceptions.NotFound):
            return render_template('404.html'), 404
            # Fallback to default behavior for other errors
        if isinstance(e, werkzeug.exceptions.Unauthorized):
            return redirect('/login')
        return super().handle_error(e)
