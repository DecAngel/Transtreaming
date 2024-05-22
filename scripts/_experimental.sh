#!/bin/bash

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/aq/aq_s_-3-2-101_mAP\=0.32130.ckpt" sap_factor=1.0 sap_strategy=aqex sap_strategy.future_length=1  sap_tag="1x Transtreaming-S on 4080 server" device_id=0
