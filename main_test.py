from db_test import *

def start(x):
    db = Company_DB()

    db.init_db()

    if x == 1:
        db.insert_data('1234', 'adasddssd')
        print("Идея записана")
    else:
        print("dsd")
    db.close()

if __name__ == "__main__":
    start(1)