from src.database import db


class User:
    def __init__(self, name):
        self.name = name
        self.sample_rate = None
        self.threshold = None
        self._create_user()

    def set_sample_rate(self, sample_rate: int):
        connection = db.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            '''
                UPDATE user
                SET sample_rate=:sample_rate
                WHERE name=:name
            ''',
            {'name': self.name, 'sample_rate': sample_rate}
        )

        connection.commit()
        connection.close()

        self.sample_rate = sample_rate

    def _create_user(self):
        # user = get_user_by_name(name)
        # if user is not None:
        #    return

        connection = db.get_connection()
        cursor = connection.cursor()

        cursor.execute(
            '''
                INSERT INTO user (name)
                VALUES (:name)
            ''',
            {'name': self.name}
        )

        connection.commit()
        connection.close()
