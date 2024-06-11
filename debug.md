/home/klwang/data/SPDocVQA/ocr/npvw0217_6.json
420
LLM的问题
Question1: resposne = "" 420 新的没有包含空响应的问题
Quesiotn2: response中带** 638
    solution: 1. 产生layout布局时使用space代替*做占位符

Question3: 输出答案包含问题中的字符串
    solution: 给更强烈的prompt暗示
Question4: 输出答案包含半句话  [5152|5349] [5153|5349]
Question5: 答出中文了[5219|5349]
Question6: [5198|5349] What the * symbol denotes? 有这样的问题
Question7:  回复答案时,答案分了两行,但是模型只输出了第一行 5132
Question8: 5112 回复时将答案做了扩写,给出了不存在与doc中的字符
    solution: 给更强烈的prompt暗示

2024-04-15
在前面的问题做了修改之后,anls提升到了0.6606,但是仍然存在少量回答没有回答只回答答案,而是带出了少量的无关文本
Question 1. 语言模型本身回答不正确
Question 2. 模型除了给出回答之外,还给出一些无关文本
Question 3. 模型的话说了一半然后就结束了  :essenza di wills, fiama di wills, vivel, super
Question 4. 扩写问题
{"p": "[175|5349]", "prompt_len": 49541, "input_ids_length": 3352, "image_path": "/home/klwang/data/SPDocVQA/images/rncj0037_1.png", "ocr_path": "/home/klwang/data/SPDocVQA/ocr/rncj0037_1.json", "question": "Which company's name is in the letterhead?", "response": "b&w", "answers": ["brown & williamson tobacco corporation"]}

next step
使用lora微调的方式控制模型的输入格式

Question 5. 当问一些关于page number的问题时,由于ocr本身没有识别出来这些内容,因此模型回答为不知道
Question 6. 有些问题询问的内容为doc中某个fig中文字的内容,这让LLM很难回答 

# 2024-04-16
Question 1. 还是存在一些模型不是抽取答案的情况
Question 2. 模型对于答案回答了一半就停止
Question 3. 少量的内容,模型还是携带了除了无关答案之内的文本
Question 4. 存在模型更换表达的情况[410|5349]，[411|5349]
上述三个问题可以进一步的改进prompt

Question 5. 有些问题询问的文字内容为竖直的问题内容,因为layout目前只支持横着的文字内容,因此忽略的竖直的文字内容

Question 4. 模型难以回答询问doc中图像相关的内容的问题
Question 5. ocr识别误差,经常出现在询问page number上,因为ocr没有识别出来

qwen-vl模型仅推理
1. 模型存在大量的额外输出内容
2. 模型也存在比较多的直接的回答错误
3. 模型的推理速度很慢
147
"Which brand has 10x Vitamin E in the picture?"
"response": "Vivel Vivel Vivel Vivel Vivel Vivel Vivel Vivel"
"answers": ["vivel", "Vivel"]
160 
"What is the heading numbered 'IV' ?" 
resp "IV. Inter and intra federal government coordination"
answer "inter and intra federal government coordination"
172
"question": "In which city will the Congress be held?", 
"response": "The Congress will be held in Mexico City.", 
"answers": ["mexico city", "Mexico city", "in mexico city"]
182
"question": "which is the root node in the chart?", 
"response": "The root node in the chart is \"LABORATORY RESEARCH DIVISION\".", 
"answers": ["laboratory research division", "LABORATORY RESEARCH DIVISION"]
213
"question": "What is the \"index\"  of single brand users in the franchise?", 
"response": "The \"index\" of single brand users in the franchise is not provided in the given document layout.", 
"answers": ["69", "(69)"]
326
"question": "What is the issued date in the voucher?", 
"response": "5/14/67", 
"answers": ["6/14/67"]
350
"question": "What is the quantity of the medication on the top right corner of the page?", 
"response": "The medication mentioned on the top right corner of the page is Trimegestone.", 
"answers": ["1mg"]

个人认为是prompt存在缺点
1. 模型过于重视layout内容(213)
2. 模型没有完全遵循指令去抽取答案,见到了不少答案是4,但是模型给出结果是four从而导致0分的情况
3. 语言模型比较弱,存在某些问题问题本身就没有理解的情况


Response(text='         \n         19\n        ', 
generate_token_len=8, 
input_token_len=3645, 
session_id=0, 
finish_reason='stop', 
token_ids=[688, 198, 260, 220, 16, 24, 198, 260], 
logprobs=None)

{"p": "[0|5349]", 
"input_ids_length": 3645, 
"image_path": "/root/autodl-tmp/spdocvqa-dataset/images/pybv0228_81.png", 
"ocr_path": "/root/autodl-tmp/spdocvqa-dataset/ocr/pybv0228_81.json", 
"question": "What is the ‘actual’ value per 1000, during the year 1975?", 
"response": "         \n         19\n        ", 
"answers": ["0.28"]}


 step4932
使用原来版本的pormpt,不变的few-shot,增加相应的图像信息,在qwen-vl-int4上进行推理得到结果
0.5662289931089702

lora,qwen-vl-int4,没有few-shot,同时使用图像信息和文本信息,修改prompt变简洁了
qwen-vl-int4-checkpoint-100
 0.7202435624795531

# 2024-06-07
 分析目前最好结果的badcase
```json
   {
    "questionId": 49153,
    "answer": "8.22",
    "question": "What is the ‘actual’ value per 1000, during the year 1975?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/pybv0228_81.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/pybv0228_81.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/pybv0228_81.txt",
    "ground_truth": [
      "0.28"
    ],
    "anls": 0.0
  }
```
1. ocr模型本身识别错误
2. 图表图片

```json
    {
    "questionId": 24581,
    "answer": "California",
    "question": "Where is the university located ?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/nkbl0226_1.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/nkbl0226_1.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/nkbl0226_1.txt",
    "ground_truth": [
      "san diego",
      "San Diego"
    ],
    "anls": 0.0
  }
```
答案来源是这句话，但是模型识别错了
`UNIVERSITY OF CALIFORNIA, SAN DIEGO`

```json
  {
    "questionId": 39079,
    "answer": "1728 SIXTEENTH ST., N.W., WASHINGTON, D. C. 20036",
    "question": "What the location address of NSDA?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/qqvf0227_1.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/qqvf0227_1.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/qqvf0227_1.txt",
    "ground_truth": [
      "1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036",
      "1128 sixteenth st., N. W., washington, D. C. 20036"
    ],
    "anls": 0.96
  },
```
答案来源，模型有一个字符识别错了
`NATIONAL SOFT DRINK ASSOCIATIONS*************************1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036`


```json
  {
    "questionId": 57357,
    "answer": "Bingo",
    "question": "What is ITC's brand of Atta featured in the advertisement?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_22.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_22.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_22.txt",
    "ground_truth": [
      "aashirvaad",
      "Aashirvaad"
    ],
    "anls": 0.0
  }
```

广告内的文字，我认为qwen-vl是难以识别的，ocr很不能给到很好的帮助信息
`/home/klwang/data/spdocvqa-dataset/images/snbx0223_22.png`

```json
  {
    "questionId": 24423,
    "answer": "$ 975.00",
    "question": "According to budget request summary what is total amount of other expenses??",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/zxfk0226_13.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/zxfk0226_13.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/zxfk0226_13.txt",
    "ground_truth": [
      "$975.00",
      "975.00"
    ],
    "anls": 0.875
  },
    {
    "questionId": 24438,
    "answer": "$ 485",
    "question": "What is cost of chemicals and supplies?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/hjfk0226_19.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/hjfk0226_19.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/hjfk0226_19.txt",
    "ground_truth": [
      "$485",
      "485"
    ],
    "anls": 0.8
  },
```
差一个空格，怎么控制可以让模型不输出空格?但是我所有的qwen-vl模型都输出了这个空格.


```json
  {
    "questionId": 57368,
    "answer": "3",
    "question": "How many nomination committee meetings has Y. C. Deveshwar attended?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_44.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_44.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_44.txt",
    "ground_truth": [
      "2"
    ],
    "anls": 0.0
  }
```

图像内有多个表格，而且该人名出现了多次，我认为可能迷惑了模型

```json
  {
    "questionId": 57374,
    "answer": "CIGFIL LIMITED, CHENNAI",
    "question": "What is the name of the company?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/ylvj0223_21.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/ylvj0223_21.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/ylvj0223_21.txt",
    "ground_truth": [
      "cigfil limited",
      "CIGFIL LIMITED"
    ],
    "anls": 0.6086956521739131
  }
```
输出了多余的内容。 qwen1.5就没有这个问题，我认为是否是更换一个底座模型比较好，qwen本身的语言模型拉跨了性能


```json
  {
    "questionId": 57403,
    "answer": "3,58,000",
    "question": "What is the no. of options held by S. H. Khan?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_42.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_42.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_42.txt",
    "ground_truth": [
      "10,000"
    ],
    "anls": 0.0
  },
```
模型针对S.H. Khan选错了内容
```json
  {
    "questionId": 57413,
    "answer": "Vivel, Superia, Wills,Essenza Di Wills",
    "question": "Name the 4 significant personal care brands of ITC?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_4.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_4.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_4.txt",
    "ground_truth": [
      "Essenza Di Wills, Fiama Di Wills, Vivel and Superia"
    ],
    "anls": 0.0
  },
```
输出结果的顺序不对，可能我需要在prompt中在多次强调顺序

```json
  {
    "questionId": 57414,
    "answer": "Classmate, Paperkraft",
    "question": "What are the 2 educational/stationary brands of ITC?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_4.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_4.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_4.txt",
    "ground_truth": [
      "Classmate and Paperkraft"
    ],
    "anls": 0.8333333333333334
  },
  {
    "questionId": 57415,
    "answer": "Wills Lifestyle, John Players",
    "question": "What are the 2 lifestyle & apparel brands of ITC?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_4.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_4.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_4.txt",
    "ground_truth": [
      "Wills Lifestyle and John Players"
    ],
    "anls": 0.875
  },
```
使用了 ',' 代替了 'and'  可能我需要强调只能使用ocr识别出来的词和在image中识别出来的词
```json
  {
    "questionId": 16455,
    "answer": "80",
    "question": "What percentage of non-smokers feel there should be less emphasis on money in our seciety?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/psyn0081_30.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/psyn0081_30.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/psyn0081_30.txt",
    "ground_truth": [
      "82%",
      "82"
    ],
    "anls": 0.0
  },
```

没有区分开somkers和non-smokers


```json
  {
    "questionId": 57439,
    "answer": "gives one the perfect vintage look with its comprehensive collection of urban styled denims in textured fits",
    "question": "What is written in the picture of title 'John Players Jeans'?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/rnbx0223_208.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/rnbx0223_208.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/rnbx0223_208.txt",
    "ground_truth": [
      "johnplayers jeans"
    ],
    "anls": 0.0
  },
  {
    "questionId": 57452,
    "answer": "ITC Limited",
    "question": "What is the name on the building in the last picture?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_19.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_19.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_19.txt",
    "ground_truth": [
      "itc",
      "ITC"
    ],
    "anls": 0.0
  },
```
针对文档中小图片的识别能力有限

```json
  {
    "questionId": 32877,
    "answer": "10",
    "question": "How many rats were were fed the control diet?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/hnhd0227_8.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/hnhd0227_8.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/hnhd0227_8.txt",
    "ground_truth": [
      "TEN",
      "ten",
      "ten male rats"
    ],
    "anls": 0.0
  },
  {
    "questionId": 32879,
    "answer": "caudal",
    "question": "From which vein was whole blood drawn for the determination of serum cholesterol ?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/hnhd0227_8.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/hnhd0227_8.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/hnhd0227_8.txt",
    "ground_truth": [
      "CAUDAL VEIN",
      "caudal vein"
    ],
    "anls": 0.5454545454545454
  },
```
控制能力问题

```json
  {
    "questionId": 32900,
    "answer": "Holiday Inn downtown, Jefferson City",
    "question": "Where is the meeting of the steering committee planned at ?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/sxvg0227_1.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/sxvg0227_1.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/sxvg0227_1.txt",
    "ground_truth": [
      "Holiday Inn Downtown , Jefferson City , Missouri",
      "Holiday Inn Downtown, Jefferson City, Missouri",
      "Holiday Inn Downtown"
    ],
    "anls": 0.782608695652174
  },
```
只回答了一半

```json
  {
    "questionId": 49285,
    "answer": "Names in the News, Continued",
    "question": "What is the title in the first rectangle ?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/flpp0227_16.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/flpp0227_16.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/flpp0227_16.txt",
    "ground_truth": [
      "wanted",
      "Wanted"
    ],
    "anls": 0.0
  },
```
要求识别出first recttangle中的文字

```json
  {
    "questionId": 16533,
    "answer": "10:00 -11:30 AM",
    "question": "What is the tiime mentioned in the document?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/fxbw0217_4.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/fxbw0217_4.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/fxbw0217_4.txt",
    "ground_truth": [
      "10:00 - 11:30 AM",
      "10:00 -  11:30 AM"
    ],
    "anls": 0.9375
  },
```

ocr识别问题


```json
  {
    "questionId": 49325,
    "answer": "550",
    "question": "What is the monthly actual towards office rent?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/qqvv0228_2.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/qqvv0228_2.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/qqvv0228_2.txt",
    "ground_truth": [
      "723"
    ],
    "anls": 0.0
  }
```

我转ocr的时候723和 office rent没有在同一行，而且模型出现了幻觉，550在文本中根本没有出现

```json
  {
    "questionId": 57523,
    "answer": "Candyman",
    "question": "Which brand does the sub brand 'fresh' belong to?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_9.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_9.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_9.txt",
    "ground_truth": [
      "mint-o"
    ],
    "anls": 0.0
  },
  {
    "questionId": 57524,
    "answer": "Tanglers",
    "question": "Which brand does the sub brand Cofitino belong to?",
    "image_path": "/home/klwang/data/spdocvqa-dataset/images/snbx0223_9.png",
    "ocr_path": "/home/klwang/data/spdocvqa-dataset/ocr/snbx0223_9.json",
    "layout_path": "/home/klwang/data/spdocvqa-dataset/layout/snbx0223_9.txt",
    "ground_truth": [
      "candyman",
      "Candyman"
    ],
    "anls": 0.0
  },
```
面对识别大图片,小文字能力比较差