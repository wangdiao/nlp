##命名实体识别NER
使用CRF标记识别实体
####训练：
python -m ner.ner --action=train

####预测：
python -m ner.ner --action=predict

####可视化
tensorboard --logdir=model/ner
