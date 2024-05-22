#!/bin/bash

Device="2080Ti server"
GPU=0

python src/sap.py experiment=longshortnet_s ckpt_path="weights/baseline/longshortnet_s_-3-2-101_29.8_official.pth" sap_factor=1.0 sap_tag="1x LongShortNet on $Device" device_id=$GPU

python src/sap.py experiment=longshortnet_s ckpt_path="weights/baseline/longshortnet_s_-3-2-101_29.8_official.pth" sap_factor=2.0 sap_tag="2x LongShortNet on $Device" device_id=$GPU

python src/sap.py experiment=longshortnet_s ckpt_path="weights/baseline/longshortnet_s_-3-2-101_29.8_official.pth" sap_factor=4.0 sap_tag="4x LongShortNet on $Device" device_id=$GPU

python src/sap.py experiment=longshortnet_s ckpt_path="weights/baseline/longshortnet_s_-3-2-101_29.8_official.pth" sap_factor=8.0 sap_tag="8x LongShortNet on $Device" device_id=$GPU

python src/sap.py experiment=longshortnet_s ckpt_path="weights/baseline/longshortnet_s_-3-2-101_29.8_official.pth" sap_factor=16.0 sap_tag="16x LongShortNet on $Device" device_id=$GPU

python src/sap.py experiment=damostreamnet_s ckpt_path="weights/baseline/damostreamnet_s_-101_31.8_official.pth" sap_factor=1.0 sap_tag="1x DAMOStreamNet-S on $Device" device_id=$GPU

python src/sap.py experiment=damostreamnet_s ckpt_path="weights/baseline/damostreamnet_s_-101_31.8_official.pth" sap_factor=2.0 sap_tag="2x DAMOStreamNet-S on $Device" device_id=$GPU

python src/sap.py experiment=damostreamnet_s ckpt_path="weights/baseline/damostreamnet_s_-101_31.8_official.pth" sap_factor=4.0 sap_tag="4x DAMOStreamNet-S on $Device" device_id=$GPU

python src/sap.py experiment=damostreamnet_s ckpt_path="weights/baseline/damostreamnet_s_-101_31.8_official.pth" sap_factor=8.0 sap_tag="8x DAMOStreamNet-S on $Device" device_id=$GPU

python src/sap.py experiment=damostreamnet_s ckpt_path="weights/baseline/damostreamnet_s_-101_31.8_official.pth" sap_factor=16.0 sap_tag="16x DAMOStreamNet-S on $Device" device_id=$GPU

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/aq/aq_s_-3-2-101_mAP\=0.32130.ckpt" sap_factor=1.0 sap_tag="1x Transtreaming-S on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/aq/aq_s_-3-2-101_mAP\=0.32130.ckpt" sap_factor=2.0 sap_tag="2x Transtreaming-S on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/aq/aq_s_-3-2-101_mAP\=0.32130.ckpt" sap_factor=4.0 sap_tag="4x Transtreaming-S on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/aq/aq_s_-3-2-101_mAP\=0.32130.ckpt" sap_factor=8.0 sap_tag="8x Transtreaming-S on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/aq/aq_s_-3-2-101_mAP\=0.32130.ckpt" sap_factor=16.0 sap_tag="16x Transtreaming-S on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/mixed/aq_drfpn_s_x2aq_mAP\=0.30722.ckpt" sap_factor=1.0 sap_tag="1x Transtreaming-S* on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/mixed/aq_drfpn_s_x2aq_mAP\=0.30722.ckpt" sap_factor=2.0 sap_tag="2x Transtreaming-S* on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/mixed/aq_drfpn_s_x2aq_mAP\=0.30722.ckpt" sap_factor=4.0 sap_tag="4x Transtreaming-S* on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/mixed/aq_drfpn_s_x2aq_mAP\=0.30722.ckpt" sap_factor=8.0 sap_tag="8x Transtreaming-S* on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1

python src/sap.py experiment=aq_drfpn_s ckpt_path="weights/trained/mixed/aq_drfpn_s_x2aq_mAP\=0.30722.ckpt" sap_factor=16.0 sap_tag="16x Transtreaming-S* on $Device" device_id=$GPU sap_strategy=aqex sap_strategy.future_length=1
