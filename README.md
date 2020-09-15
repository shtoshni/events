# events

Data processing
```
python data_processing/xml_to_conll.py ../data/red/ ../proc_data/red/independent/ 
```
To add the duplicate tokens to the doc, add the -add_duplicate_tag flag as:
```
python data_processing/xml_to_conll.py ../data/red/ ../proc_data/red/independent_duplicate/ -add_duplicate_tag 
```

Training/Testing the model:
```
python auto_memory_model/main.py  -doc_enc independent -model_size base  -pretrained_bert_dir /share/data/speech/shtoshni/resources -seed 25 -finetune  -mlp_size 1024 -sample_singletons 0.2  -proc_strategy default   -ft_lr 3e-05  -focus_group entity  -no_singleton
```
Flag info:
* -finetune: Finetunes the document encoder i.e. BERT in this case. Otherwise no finetuning.
* -sample_singletons 0.2: Determines the fraction of singletons to be sampled during training.
* -ft_lr 3e-5: Fine-tuning learning rate
* -proc_strategy {default/duplicate}: Whether duplicate tags are added or not to the text.  
* -focus_group {entity/event/joint}: If entity or event, just train and eval on the specified focus group. Otherwise do joint training and evaluation.
* -no_singleton: Ensures that during evaluation we don't consider singleton clusters during metric evaluation.
 
 

