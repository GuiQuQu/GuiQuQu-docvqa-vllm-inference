
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

