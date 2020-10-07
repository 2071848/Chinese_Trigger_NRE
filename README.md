# Chinese_Trigger_NRE
## Requirements
#### Python 3.6
#### Pytorch 0.4.1
## 具体用法可以参考
#### https://github.com/thunlp/Chinese_NRE
## ACE data parse
##### 可以通过ace_data_parse得到如下形式的句子：
###### head  tail  relation  sentences
###### 举例：应变小组 	警网就近的万芳医院	 PART-WHOLE/Subsidiary	 捷运公司接到报案立刻通知警网就近的万芳医院成立应变小组，将伤患迅速送往急诊室
#### ace_index.txt是数据集中句子为显式类型、蕴含类型和推理类型的index：
###### 举例：第一列表示ACE2005中文数据集中的关系ID，第二列如果是数字表示在这个句子中这个位置的词是触发词（位置从0开始），如果是<infer>表示句子是推理类型，如果是<omit>表示是蕴含类型；
###### XIN20001013.0200.0027-R13-1       	22
###### CNR20001208.1700.1129-R10-1  <infer>
###### CNR20001201.1700.1429-R2-1 <omit>

## /data/pretrained_model
###### 请到bert官网下载预训练的BERT模型
##### BERT-Base, Chinese：https://github.com/google-research/bert
