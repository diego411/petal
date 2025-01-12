import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler

from flask import Flask, render_template
from flask_socketio import SocketIO
from src.PlantApi import PlantApi

from src.AppConfig import AppConfig
from src.controller import dropbox_controller
from src.database import db

from src.resource.api.RecordingResouce import RecordingResource
from src.resource.api.RecordingActionResource import RecordingActionResource
from src.resource.template.TemplateResource import TemplateResource
from src.resource.template.EntityTemplateResource import EntityTemplateResource
from src.resource.api.LegacyResource import LegacyResource

socketio = SocketIO()


def create_app():
    app = Flask(__name__)
    app.config.from_object(AppConfig)

    socketio.init_app(app)
    app.socketio = socketio

    db.init_tables()
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

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('404.html'), 404

    API_ROUTE = f'/api/{app.config.get("API_VERSION")}'
    api = PlantApi(app)

    api.add_resource(
        RecordingResource,
        f"{API_ROUTE}/recording",
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
        TemplateResource,
        '/',
        '/<string:template>',
    )

    api.add_resource(
        EntityTemplateResource,
        '/<string:template>/<string:entity_id>',
    )

    @app.context_processor
    def inject_version():
        return dict(version=app.config['VERSION'])

    return app


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(debug=True)
    socketio.run(flask_app, host="0.0.0.0", port=5000)
