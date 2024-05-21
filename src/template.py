
# 使用*做占位符的模板
star_question_templatev1 = """The following is a string layout of a document picture, including by "```", and in the layout, I use "*" as the placeholder,the text before the placeholder should fill the all space in the actual picture.
Please answer the question according to the layout infomation.
Document:
```
{layout}
```
Question: {question}
Answer:"""

star_question_templatev2 = """The following is a string layout of a document picture, including by "```", and in the layout, I use "*" as placeholder, the text segment should fill not only the space of the text, but also the space of '*' segment when a text segment is following by the segment of '*'.
Notice:
(1) You should answer the qeuistion based on the layout information, using the words appeared in the layout is best.
(2) Please DO NOT appear the '*' in your answer.
(3) Answer the question directly! Please DO NOT repeat the question text in your answer, only privide the answer of question.
Document:
```
{layout}
```
Question: {question}
Answer:"""

star_question_templatev3 = """The following is a string layout of a document picture, including by "```", and in the layout, I use "*" as placeholder, the text segment should fill not only the space of the text, but also the space of '*' segment when a text segment is following by the segment of '*'.
Please extract the answer of the question from the given document layout.
Notice:
(1) You should answer the qeuistion based on the layout information, using the words appeared in the layout is best.
(2) Please DO NOT appear the '*' in your answer.
(3) Answer the question directly! Please DO NOT repeat the question text in your answer, only privide the answer of question.
Document:
```
{layout}
```
Question: {question}
Answer:"""

star_question_templatev4 = """The following is a string layout of a document picture, including by "```".
Notice:
(1) You should answer the qeuistion based on the layout information, using the words appeared in the layout is best.
(2) Please DO NOT appear the '*' in your answer.
(3) Please extract the answer of the question from the given document layout.
Document:
```
{layout}
```
Question: {question}
Answer:"""

star_question_template_with_img = """The following is a document image and its corresponding string layout of a document picture, including by "```", and in the layout, I use "*" as placeholder, the text segment should fill not only the space of the text, but also the space of '*' segment when a text segment is following by the segment of '*'.
Please extract the answer of the question from the given document layout.
Notice:
(1) You should answer the qeuistion based on the layout information, using the words appeared in the layout is best.
(2) Please DO NOT appear the '*' in your answer.
(3) Answer the question directly! Please DO NOT repeat the question text in your answer, only privide the answer of question.
Image:
```
{image_path}
```
Document:
```
{layout}
```
Question: {question}
Answer:"""

"""
<|im_start|>\nnYou are a helpful assistant.<|im_end|>\nuser\nPicture 0:{image_path}\n
"""


vl_ocr_question_template = """The following is a document image and its corresponding string layout of a document picture, including by "```".Please extract the answer from the document layout based on the document image.

Document Layout Notice:
In the layout, "*" is used as the placeholder, this meaning the text segment should fill text part and "*" part in document picture.

Document Picture Notice:
If you don't find answer from the document layout,please answer the question based on the picture.

Notice:
(1) Answer the question based on the layout and image, and answer the question using text that appears in layout and image.
(2) Please DO NOT appear the '*' in your answer
Document Picture:
```
{image_path}
```
Document Layout:
```
{layout}
```
Question: {question}
Answer:"""

lmdeploy_vl_ocr_question_template = """Integrating the document image and text layout infomation, please answer the question. layout will be included by "```", image will start with "Picture 0"
Notice:
(1) answer the question based on the layout and image, and answer the question using text that appears in layout and image.
(2) Please DO NOT appear the '*' in your answer
Layout:
```
{layout}
```
Question: {question}
Answer:"""


hat_question_templatev1 = """The following is a string layout of a document picture, including by "```", and in the layout, I use "^" as placeholder, the text segment should fill not only the space of the text, but also the space of '^' segment when a text segment is following by the segment of '^'.
Notice:
(1) You should answer the qeuistion based on the layout information, using the words appeared in the layout is best.
(2) Please DO NOT appear the '^' in your answer.
(3) Answer the question directly! Please DO NOT repeat the question text in your answer, only privide the answer of question.

Document:
```
{layout}
```
Question: {question}
Answer:"""

# 使用" "做占位符的模板
space_question_template = """The following is a string layout of a document picture, including by "```". Please answer the question according to the layout infomation.
Document:
```
{layout}
```
Question: {question}
Answer:"""

