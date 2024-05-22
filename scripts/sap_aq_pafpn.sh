#!/bin/bash

Device="2080Ti server"
GPU=0

python src/sap.py experiment=aq_pafpn_s ckpt_path="weights/trained/mixed/aq_pafpn_s_x2aq_mAP\=0.28337.ckpt" sap_factor=1.0 sap_tag="1x Transtreaming-S** on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_pafpn_s ckpt_path="weights/trained/mixed/aq_pafpn_s_x2aq_mAP\=0.28337.ckpt" sap_factor=2.0 sap_tag="2x Transtreaming-S** on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_pafpn_s ckpt_path="weights/trained/mixed/aq_pafpn_s_x2aq_mAP\=0.28337.ckpt" sap_factor=4.0 sap_tag="4x Transtreaming-S** on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_pafpn_s ckpt_path="weights/trained/mixed/aq_pafpn_s_x2aq_mAP\=0.28337.ckpt" sap_factor=8.0 sap_tag="8x Transtreaming-S** on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_pafpn_s ckpt_path="weights/trained/mixed/aq_pafpn_s_x2aq_mAP\=0.28337.ckpt" sap_factor=16.0 sap_tag="16x Transtreaming-S** on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1
