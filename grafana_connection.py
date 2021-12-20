import json
import random
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

clientSmartNetwork = mqtt.Client()


def send_init_message():
    clientSmartNetwork.publish("node/init", json.dumps(
        {
            "type": "internet",
            "id": "test_prediction",
            "name": "test_prediction",
            "measurements": [{
                "name": "internal_climate",
                "description": "predicted internal climate",
                "unit": "Degree"
            }],
            "actuators": [{}],
        }))


def process_test_data():
    clientSmartNetwork.publish("node/data", json.dumps({
        "id": "test_prediction",
        "measurements": [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "internal_climate": random.randrange(20, 30) * 1.0
        }]
    }))


if __name__ == '__main__':
    clientSmartNetwork.username_pw_set("node", password="smartmeternode")
    clientSmartNetwork.connect("sendlab.nl", 11884, 60)

    # send_init_message()

    process_test_data()

