import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler

from flask import Flask, render_template, redirect, url_for
from flask_socketio import SocketIO
from src.PlantApi import PlantApi

from src.AppConfig import AppConfig
from src.controller import dropbox_controller
from src.database import db

from src.resource.api.RecordingResouce import RecordingResource
from src.resource.api.RecordingActionResource import RecordingActionResource
from src.resource.api.ExperimentResource import ExperimentResource
from src.resource.api.ExperimentActionResource import ExperimentActionResource
from src.resource.api.ObservationResource import ObservationResource
from src.resource.template.TemplateResource import TemplateResource
from src.resource.template.NestedTemplateResource import NestedTemplateResource
from src.resource.template.EntityTemplateResource import EntityTemplateResource
from src.resource.api.LegacyResource import LegacyResource
from src.resource.api.LogoutResource import LogoutResource
from src.resource.api.LoginResource import LoginResource
from src.entity.exception.UnauthorizedException import UnauthorizedException

socketio = SocketIO()


def create_app():
    app = Flask(__name__)
    app.config.from_object(AppConfig)

    socketio.init_app(app)
    app.socketio = socketio

    db.run_migrations()
    db.create_measurement_partition()
    db.create_measurement_partition(offset=1)

    dropbox_client = dropbox_controller.create_dropbox_client(
        app_key=app.config['DROPBOX_APP_KEY'],
        app_secret=app.config['DROPBOX_APP_SECRET'],
        refresh_token=app.config['DROPBOX_REFRESH_TOKEN']
    )
    app.dropbox_client = dropbox_client

    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_file = os.path.join('logs', 'app.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,  # Log level can be changed as needed
        format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Set log level
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    file_handler.setFormatter(formatter)

    app.logger.addHandler(file_handler)

    scheduler = BackgroundScheduler()
    scheduler.add_job(db.create_measurement_partition, 'cron', args=[1], hour=23, minute=52)
    scheduler.start()

    API_ROUTE = f'/api/{app.config.get("API_VERSION")}'
    api = PlantApi(app)

    api.add_resource(
        LoginResource,
        f"{API_ROUTE}/login"
    )

    api.add_resource(
        LogoutResource,
        f"{API_ROUTE}/logout"
    )

    api.add_resource(
        RecordingResource,
        f"{API_ROUTE}/recording",
        f"{API_ROUTE}/recordings",
        f"{API_ROUTE}/recording/<string:recording_id>",
        endpoint="recording"
    )

    api.add_resource(
        RecordingActionResource,
        f"{API_ROUTE}/recording/<string:recording_id>/<string:action>",
        endpoint="recording_action"
    )

    api.add_resource(
        LegacyResource,
        f'{API_ROUTE}/legacy/<string:action>'
    )

    api.add_resource(
        ExperimentResource,
        f"{API_ROUTE}/experiment",
        f"{API_ROUTE}/experiment/<string:experiment_id>",
        endpoint="experiment"
    )

    api.add_resource(
        ExperimentActionResource,
        f"{API_ROUTE}/experiment/<string:experiment_id>/<string:action>",
        endpoint="experiment_action"
    )

    api.add_resource(
        ObservationResource,
        f"{API_ROUTE}/observation",
        endpoint="observation"
    )

    api.add_resource(
        TemplateResource,
        '/',
        '/<string:template>',
    )

    api.add_resource(
        NestedTemplateResource,
        '/<string:module>/<string:template>'
    )

    api.add_resource(
        EntityTemplateResource,
        '/<string:template>/<int:entity_id>',
    )

    @app.context_processor
    def inject_version():
        return dict(version=app.config['VERSION'])

    @app.route('/index', methods=['GET'])
    def index():
        return render_template('index.html')

    @app.route('/login', methods=['GET'])
    def login_template():
        return render_template('login.html')

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template(
            'error.html',
            message='404 - Page Not Found'
        ), 404

    @app.errorhandler(401)
    def not_authorized():
        return redirect(url_for('login')), 401

    @app.errorhandler(UnauthorizedException)
    def custom_unauthorized(error: UnauthorizedException):
        if error.origin == 'api':
            return error.message, 401
        elif error.origin == 'template':
            return render_template(
                'error.html',
                message=error.message
            ), 401

    return app


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(debug=True)
    socketio.run(flask_app, host="0.0.0.0", port=5000)
