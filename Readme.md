**Repository for MSc Project "Evaluations of Large Language Models using the Abstraction and Reasoning Corpus"**  
City St George's - University of London  
Grigorios Vaitsas  
1 November 2024  

Directories in this repository and their usage:  

1) **datasets** Directory  
Contains the whole ARC dataset in the original format under 2020 dataset/  
or the format used in the ARC-Prize competition 2024 under arc-prize-2024/  

The 50 ARC tasks used in this analysis can be found under subset50/ as individual .json files and also in the files  
50_word_in_quotes_challenges.json and 50_word_in_quotes_solutions.json  
containing all 50 challenges and their corresponding 50 solutions respectively  

The images of the tasks used to prompt multimodal LLMs are found under vision_dataset/  

2) **text_encoding_scripts** Directory  
Contains the scripts used for the direct-grid encoding tests on the LLMs as well as the tests used for the   
Chain of Thought experiments. Direct-grid tests have few-shot prefix and CoT tests have cot prefix.  

The naming explains the kind of test performed by the script so for example:  
few_shot_test_plain_nums.py executes the test where the grids are represented by numbers in compact format and  
few_shot_test_pipe_words.py executes the test where the grids are represented by strings separated by pipe |  

Similarly   
cot_few_shot_test_json_nums_with_example.py is a CoT test where json numbers are used in a CoT that uses example  
and   
cot_few_shot_test_json_words_extended_example.py is a CoT where json strings are used in a CoT test with extended example  

Code is executed by typing:  
**python <script_name>**  
for example:  
python few_shot_test_plain_nums.py  
The user will be prompted to select the model to test against as well as the number of tasks to use:  

Select the model to use:  
1: GPT-3.5 (gpt-3.5-turbo)  
2: GPT-4o Mini (gpt-4o-mini)  
3: GPT-4o (gpt-4o)  
4: o1 Mini (o1-mini)  
5: o1 Preview (o1-preview)  
6: Claude-3.5 Sonnet (claude-3-5-sonnet-20240620)  
Enter the number corresponding to the model (1 to 6):   
Enter the number of tasks you want to process (or 'all' to process all tasks):  

3) **object_encoding_scripts** Directory  
Contains the scripts used for object-based encoding tests. These consist of 2 scripts:  
abstraction.py contains the list of abstractions that can be acted on a given task to generate a graph  
This code has been adapted from  
https://github.com/khalil-research/ARGA-AAAI23/  
main.py executes the tests on the LLMs once the correct abstraction has been applied, receives the response  
and scores the result. Again portions of the code have been adapted from  
https://github.com/khalil-research/ARGA-AAAI23/  

abstractions.txt indicates the type of abstraction used by each task in our subset of 50 tasks  

ARGA-solutions Directory contains the solutions files that are produced when running the ARGA tool   
on the subset of 50 tasks. We use these .json files to extract the type of abstraction that is best  
suited for each of our task while loading the tasks.   

Code is executed by typing:   
**python main.py**  

Then the user will be promptes to choose the model they want to test against as well as the encoding used for the task  
and the number of tasks:  

Select the model to use:  
1: GPT-3.5 (gpt-3.5-turbo)  
2: GPT-4o Mini (gpt-4o-mini)  
3: GPT-4o (gpt-4o)  
4: o1 Mini (o1-mini)  
5: o1 Preview (o1-preview)  
6: Claude-3.5 Sonnet (claude-3-5-sonnet-20240620)  
Enter the number corresponding to the model (1 to 6):   
Please select the encoding type:  
1) Object JSON  
2) Object Descriptor  
3) Object JSON with edge  
4) Object Descriptor with edge  
5) Object JSON with words  
Select 1-5:  
Enter the number of tasks you want to process (or 'all' to process all tasks):  

4) **log_txt_output** Directory  
Contains the logs of the direct-grid encoding tests. These are split into:  
   few_shot directory (No Cot used) and the  
   cot_few_shot when CoT was used  

5) **log_obj_output** Directory  
Contains the logs of all the test that were performed using the object-based encodings.   
